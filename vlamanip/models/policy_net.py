"""Language-conditioned pick-and-place network.

Inputs (all from the camera and the instruction):
    maps      (B, 4, H, W)  top-down colour (3) and height (1) maps, normalised
    clip_img  (B, Dv, 14, 14) CLIP patch features of the colour map            (semantic pathway)
    words     (B, L, Dt) CLIP token features, ``word_mask`` (B, L)              (language pathway)
    sentence  (B, E) CLIP sentence embedding

Architecture:
    spatial encoder   residual CNN on the maps, 160 -> 80 -> 40 -> 20 (skip features kept)
    fusion            at 20 x 20: spatial + upsampled CLIP patches -> tokens; each head runs transformer blocks
                      (self-attention over the map, cross-attention to the instruction words)
    decoders          one per head (pick, place): upsampling with skip connections back to 160 x 160, every stage
                      modulated by the sentence embedding (FiLM)
    outputs           pick and place logits (B, R, H, W): R yaw bins x every map cell

Pick and place have separate fusion and decoder weights so each can attend to its own part of the sentence
(the object for picking, the destination for placing).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm2d(c: int) -> nn.BatchNorm2d:
    # BatchNorm is ~2x faster than GroupNorm here (bfloat16, channels_last); batches are 32 during training
    return nn.BatchNorm2d(c)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = norm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = norm2d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm2d(cout))

    def forward(self, x):
        y = F.relu(self.n1(self.conv1(x)))
        y = self.n2(self.conv2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class FiLM(nn.Module):
    """Feature-wise affine modulation from a conditioning vector (initialised to the identity)."""

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, cond):
        gamma, beta = self.lin(cond).chunk(2, dim=-1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]


class FusionBlock(nn.Module):
    """Self-attention over map tokens, cross-attention to word tokens, feed-forward (pre-norm)."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, words, word_pad):
        h = self.n1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        x = x + self.cross_attn(self.n2(x), words, words, key_padding_mask=word_pad, need_weights=False)[0]
        return x + self.ffn(self.n3(x))


class Head(nn.Module):
    """Fusion + decoder for one output (pick or place)."""

    def __init__(self, dim: int, skip_channels: tuple[int, int, int], cond_dim: int, num_rotations: int, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(FusionBlock(dim) for _ in range(depth))
        c3, c2, c1 = skip_channels
        # narrow decoder: most of the compute is at 80^2 and 160^2, where only local detail is refined
        self.dec3, self.film3 = ResBlock(dim + c3, 128), FiLM(cond_dim, 128)
        self.dec2, self.film2 = ResBlock(128 + c2, 64), FiLM(cond_dim, 64)
        self.dec1, self.film1 = ResBlock(64 + c1, 32), FiLM(cond_dim, 32)
        self.out = nn.Conv2d(32, num_rotations, 1)

    def forward(self, tokens, size, words, word_pad, sentence, skips):
        for blk in self.blocks:
            tokens = blk(tokens, words, word_pad)
        b, _, d = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, d, *size)
        s3, s2, s1 = skips
        x = F.interpolate(x, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.film3(self.dec3(torch.cat((x, s3), 1)), sentence)
        x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.film2(self.dec2(torch.cat((x, s2), 1)), sentence)
        x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.film1(self.dec1(torch.cat((x, s1), 1)), sentence)
        return self.out(x)


class PickPlaceNet(nn.Module):
    def __init__(self, clip_image_dim: int = 768, clip_text_dim: int = 512, clip_embed_dim: int = 512,
                 base_channels: int = 64, dim: int = 256, num_rotations: int = 8, fusion_depth: int = 2, map_size: int = 160):
        super().__init__()
        c = base_channels
        self.stem = nn.Sequential(nn.Conv2d(4, c, 3, 1, 1, bias=False), norm2d(c), nn.ReLU())
        self.enc1 = ResBlock(c, c)                # 160
        self.enc2 = ResBlock(c, 2 * c, 2)         # 80
        self.enc3 = ResBlock(2 * c, 4 * c, 2)     # 40
        self.enc4 = ResBlock(4 * c, dim, 2)       # 20
        self.sem = nn.Sequential(nn.Conv2d(clip_image_dim, dim, 1), nn.GELU())
        self.merge = nn.Conv2d(2 * dim, dim, 1)
        self.word_proj = nn.Linear(clip_text_dim, dim)
        side = map_size // 8
        self.pos = nn.Parameter(torch.zeros(1, side * side, dim))
        nn.init.normal_(self.pos, std=0.02)
        skips = (4 * c, 2 * c, c)
        self.pick_head = Head(dim, skips, clip_embed_dim, num_rotations, fusion_depth)
        self.place_head = Head(dim, skips, clip_embed_dim, num_rotations, fusion_depth)

    def forward(self, maps, clip_img, words, word_mask, sentence):
        s1 = self.enc1(self.stem(maps))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        sem = F.interpolate(self.sem(clip_img.to(s4.dtype)), size=s4.shape[-2:], mode="bilinear", align_corners=False)
        x = self.merge(torch.cat((s4, sem), 1))
        b, d, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2) + self.pos
        words = self.word_proj(words.to(tokens.dtype))
        pad = ~word_mask
        sentence = sentence.to(tokens.dtype)
        skips = (s3, s2, s1)
        pick = self.pick_head(tokens, (h, w), words, pad, sentence, skips)
        place = self.place_head(tokens, (h, w), words, pad, sentence, skips)
        return pick, place
