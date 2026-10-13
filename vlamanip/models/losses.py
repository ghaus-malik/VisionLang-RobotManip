"""Soft-label cross-entropy over (yaw bin x map cell) logits and arg-max decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_targets(cells: torch.Tensor, num_rotations: int, height: int, width: int, sigma_px: float,
                 sigma_rot: float = 0.5) -> torch.Tensor:
    """(B, 3) label cells -> (B, R, H, W) target distributions.

    Gaussian over the map around the labelled cell (``sigma_px``) times a circular Gaussian over yaw bins
    (``sigma_rot`` bins; bin R-1 is next to bin 0 because yaw wraps every 90 degrees).
    """
    dev = cells.device
    rows = torch.arange(height, device=dev).float()
    cols = torch.arange(width, device=dev).float()
    gr = torch.exp(-0.5 * ((rows[None] - cells[:, 1:2].float()) / sigma_px) ** 2)  # (B, H)
    gc = torch.exp(-0.5 * ((cols[None] - cells[:, 2:3].float()) / sigma_px) ** 2)  # (B, W)
    k = torch.arange(num_rotations, device=dev).float()
    d = (k[None] - cells[:, 0:1].float()).remainder(num_rotations)
    d = torch.minimum(d, num_rotations - d)
    gk = torch.exp(-0.5 * (d / sigma_rot) ** 2)  # (B, R)
    t = gk[:, :, None, None] * gr[:, None, :, None] * gc[:, None, None, :]
    return t / t.sum(dim=(1, 2, 3), keepdim=True)


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of -sum(target * log_softmax(logits)) over all R*H*W outputs."""
    logp = F.log_softmax(logits.float().flatten(1), dim=1)
    return -(targets.flatten(1) * logp).sum(1).mean()


def argmax_cells(logits: torch.Tensor) -> torch.Tensor:
    """(B, R, H, W) logits -> (B, 3) best (rotation bin, row, col)."""
    b, r, h, w = logits.shape
    idx = logits.flatten(1).argmax(1)
    return torch.stack((idx // (h * w), (idx // w) % h, idx % w), dim=1)
