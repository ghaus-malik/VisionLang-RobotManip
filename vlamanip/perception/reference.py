"""Ground-truth top-down height raster from object poses (used only for testing the camera pipeline and metrics).

The learned policy never sees this map; it exists to check that the camera-built map is correct and to measure
how much of the workspace the arm hides in a given pose.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from vlamanip.geometry import WorkspaceGrid


def reference_heightmap(
    cube_poses: dict[str, tuple[float, float, float, float]],
    bin_positions: dict[str, tuple[float, float]],
    scene_cfg,
    grid: WorkspaceGrid,
) -> torch.Tensor:
    """(H, W) height above the table of every cell centre given cube poses (x, y, z, yaw) and bin positions (x, y)."""
    rows, cols = torch.meshgrid(torch.arange(grid.height), torch.arange(grid.width), indexing="ij")
    xy = grid.pixel_to_world(torch.stack((rows, cols), dim=-1))
    height = torch.zeros(grid.height, grid.width)

    inner_half = scene_cfg.bin_inner / 2
    outer_half = inner_half + scene_cfg.bin_wall_thickness
    for bx, by in bin_positions.values():
        d = torch.maximum((xy[..., 0] - bx).abs(), (xy[..., 1] - by).abs())  # bins are axis-aligned
        height = torch.where(d < outer_half, torch.maximum(height, torch.tensor(scene_cfg.bin_floor_thickness)), height)
        height = torch.where((d < outer_half) & (d >= inner_half), torch.maximum(height, torch.tensor(scene_cfg.bin_wall_height)), height)

    half = scene_cfg.cube_size / 2
    for x, y, z, yaw in cube_poses.values():
        dx, dy = xy[..., 0] - x, xy[..., 1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        u, v = c * dx + s * dy, -s * dx + c * dy  # cell centre in the cube frame
        inside = (u.abs() < half) & (v.abs() < half)
        height = torch.where(inside, torch.maximum(height, torch.tensor(z + half)), height)
    return height


def map_error_mask(height: torch.Tensor, reference: torch.Tensor, tolerance: float = 0.01) -> torch.Tensor:
    """Cells whose measured height is outside the reference range of their 3x3 neighbourhood by more than ``tolerance``.

    The neighbourhood range forgives the one-cell blur at object edges; occluders (the arm) and projection errors
    still show up.
    """
    ref = reference[None, None]
    hi = F.max_pool2d(ref, 3, stride=1, padding=1)[0, 0]
    lo = -F.max_pool2d(-ref, 3, stride=1, padding=1)[0, 0]
    return (height > hi + tolerance) | (height < lo - tolerance)
