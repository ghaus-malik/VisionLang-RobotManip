"""Frozen CLIP encoders (Hugging Face ``transformers``) for the semantic pathway and the language pathway.

* Image: the top-down colour map is resized to 224 px and encoded by the CLIP vision transformer; the patch tokens
  (B, 768, 14, 14) give a coarse semantic feature map aligned with the workspace.
* Text: the instruction is encoded by the CLIP text transformer; the token features (B, L, 512) let the policy
  attend to individual words (which colour is the object, which is the destination) and the projected sentence
  embedding (B, 512) conditions the decoders.

The encoders are never trained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FrozenCLIP(nn.Module):
    def __init__(self, name: str = "openai/clip-vit-base-patch16", image_size: int = 224, max_text_length: int = 32):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.model = CLIPModel.from_pretrained(name)
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.model.eval().requires_grad_(False)
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    @property
    def image_dim(self) -> int:
        return self.model.config.vision_config.hidden_size

    @property
    def text_dim(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self.model.config.projection_dim

    def train(self, mode: bool = True):  # always stays in eval mode
        return super().train(False)

    @torch.no_grad()
    def encode_image(self, rgb: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) colours in [0, 1] -> (B, D, 14, 14) patch features."""
        x = F.interpolate(rgb.float(), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        vision = self.model.vision_model
        tokens = vision.post_layernorm(vision(pixel_values=x).last_hidden_state[:, 1:])  # drop the class token
        b, n, d = tokens.shape
        side = int(round(n ** 0.5))
        return tokens.transpose(1, 2).reshape(b, d, side, side)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Instructions -> token features (B, L, Dt), valid-token mask (B, L) and sentence embedding (B, E)."""
        enc = self.tokenizer(texts, padding="max_length", max_length=self.max_text_length, truncation=True, return_tensors="pt")
        dev = self.mean.device
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev)
        out = self.model.text_model(input_ids=ids, attention_mask=mask)
        sentence = self.model.text_projection(out.pooler_output)
        return out.last_hidden_state, mask.bool(), sentence
