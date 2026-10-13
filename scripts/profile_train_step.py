"""Time the parts of one training step (data, augmentation, CLIP, network forward/backward) on the GPU."""

from __future__ import annotations

import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.data.augment import augment  # noqa: E402
from vlamanip.models.clip_encoder import FrozenCLIP  # noqa: E402
from vlamanip.models.losses import soft_cross_entropy, soft_targets  # noqa: E402
from vlamanip.policy import build_net  # noqa: E402


def timed(name, fn, reps=10):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    print(f"{name:28s} {(time.time() - t) / reps * 1000:7.1f} ms", flush=True)
    return out


def main():
    torch.backends.cudnn.benchmark = True
    cfg = load_config()
    dev, b = "cuda:0", cfg.train.batch_size
    clip = FrozenCLIP(cfg.model.clip_name).to(dev)
    net = build_net(cfg.model, clip, 8, 160).to(dev).to(memory_format=torch.channels_last)
    maps = torch.randn(b, 4, 160, 160, device=dev).contiguous(memory_format=torch.channels_last)
    cells = torch.randint(0, 160, (b, 3), device=dev)
    cells[:, 0] %= 8
    words, mask, sent = clip.encode_text(["put the red cube in the blue bin"] * b)

    timed("augment", lambda: augment(maps, [cells, cells], 8, True, True, 8))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        img = timed("clip image", lambda: clip.encode_image(maps[:, :3]))
        timed("net forward (no grad)", lambda: torch.no_grad()(net)(maps, img, words, mask, sent))
    tgt = timed("soft targets x2", lambda: (soft_targets(cells, 8, 160, 160, 1.5), soft_targets(cells, 8, 160, 160, 2.0)))

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p, q = net(maps, img, words, mask, sent)
        loss = soft_cross_entropy(p, tgt[0]) + soft_cross_entropy(q, tgt[1])
        loss.backward()

    timed("net forward + backward", step)
    print(f"max memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

    print(f"trainable parameters {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M")


if __name__ == "__main__":
    main()
