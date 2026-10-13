"""Tabletop layouts: where each cube and bin starts (pure NumPy, no simulator).

A layout stores, per colour, the cube pose (x, y, yaw) and the bin position (x, y); bins are axis-aligned.
Objects are placed by rejection sampling with axis-aligned square footprints (a bin's outer square; for a cube
the square bounding it at any yaw), so that

* bins never overlap (``gap_bin`` free space between them),
* every cube has ``gap_cube`` free space around it, enough for the open gripper fingers,
* cubes start on the table, never inside a bin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Layout:
    cubes: dict[str, tuple[float, float, float]] = field(default_factory=dict)  # colour -> (x, y, yaw)
    bins: dict[str, tuple[float, float]] = field(default_factory=dict)  # colour -> (x, y)

    def to_dict(self) -> dict:
        return {"cubes": {k: list(v) for k, v in self.cubes.items()}, "bins": {k: list(v) for k, v in self.bins.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls({k: tuple(v) for k, v in d["cubes"].items()}, {k: tuple(v) for k, v in d["bins"].items()})


def cube_half_extent(scene_cfg) -> float:
    """Half side of the axis-aligned square that contains the cube at any yaw."""
    return scene_cfg.cube_size * math.sqrt(2) / 2


def bin_half_extent(scene_cfg) -> float:
    return (scene_cfg.bin_inner + 2 * scene_cfg.bin_wall_thickness) / 2


def square_gap(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Free space between two axis-aligned squares (x, y, half_extent); negative if they overlap."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) - (a[2] + b[2])


def sample_layout(
    rng: np.random.Generator,
    scene_cfg,
    layout_cfg,
    cube_colors: list[str],
    bin_colors: list[str],
    region_x: tuple[float, float] | None = None,
    region_y: tuple[float, float] | None = None,
) -> Layout:
    """Random non-overlapping layout; bins are placed first (they are the largest objects).

    ``region_x`` / ``region_y`` bound the object centres.
    """
    rx = region_x or tuple(layout_cfg.region_x)
    ry = region_y or tuple(layout_cfg.region_y)
    h_cube, h_bin = cube_half_extent(scene_cfg), bin_half_extent(scene_cfg)
    for _ in range(layout_cfg.max_tries):
        placed: list[tuple[float, float, float, float]] = []  # (x, y, half_extent, gap)
        layout = Layout()
        ok = True
        for color in bin_colors:
            p = _place(rng, rx, ry, h_bin, layout_cfg.gap_bin, placed)
            if p is None:
                ok = False
                break
            layout.bins[color] = p
        if not ok:
            continue
        for color in cube_colors:
            p = _place(rng, rx, ry, h_cube, layout_cfg.gap_cube, placed)
            if p is None:
                ok = False
                break
            layout.cubes[color] = (p[0], p[1], float(rng.uniform(0.0, math.pi / 2)))
        if ok:
            return layout
    raise RuntimeError("Could not sample a layout; enlarge the region or reduce the gaps.")


def _place(rng, rx, ry, half, gap, placed, tries: int = 200):
    """One position with at least ``gap`` (and the other object's gap) free space to every placed object."""
    for _ in range(tries):
        x, y = float(rng.uniform(*rx)), float(rng.uniform(*ry))
        if all(square_gap((x, y, half), (px, py, ph)) >= max(gap, pg) for px, py, ph, pg in placed):
            placed.append((x, y, half, gap))
            return (x, y)
    return None


def fixed_layout(stage: int) -> Layout:
    """Hand-made layouts for debugging (stage 1: one cube and one bin)."""
    if stage == 1:
        return Layout(cubes={"blue": (0.45, 0.12, 0.0)}, bins={"red": (0.55, -0.12)})
    return Layout(
        cubes={"red": (0.38, 0.15, 0.0), "blue": (0.38, 0.05, 0.3), "green": (0.38, -0.05, 0.6), "yellow": (0.38, -0.15, 0.9)},
        bins={"red": (0.60, 0.17), "blue": (0.60, 0.05), "green": (0.60, -0.07), "yellow": (0.60, -0.19)},
    )
