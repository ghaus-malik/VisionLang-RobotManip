"""Scripted expert: pick and place poses computed from ground-truth object poses.

Used only to generate demonstrations (and as an upper-bound reference). The learned policy never calls it.

Poses are top-down gripper poses ``(x, y, z_tcp, yaw)``: z_tcp is the height of the point between the fingertips
and yaw the rotation of the hand about world +z (modulo 90 deg). Each action also has a *support height*, the
height of the surface under the target pixel (cube top for picking; bin floor or cube top for placing). The
learned policy reads the same quantity from its height map, so expert and policy share :func:`tcp_height_pick`
and :func:`tcp_height_place`.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from vlamanip.language.templates import PICK_PLACE, STACK, TaskSpec


@dataclass
class Action:
    x: float
    y: float
    yaw: float  # radians in [0, pi/2)
    support_height: float  # height of the surface at (x, y)
    z_tcp: float

    def to_dict(self) -> dict:
        return asdict(self)


def tcp_height_pick(top_height: float, control_cfg) -> float:
    """Close the fingers ``grasp_depth`` below the top face of the cube."""
    return top_height - control_cfg.grasp_depth


def tcp_height_place(support_height: float, scene_cfg, control_cfg) -> float:
    """Release with the bottom of the held cube ``place_clearance`` above the support surface.

    The cube is held with its top face ``grasp_depth`` above the TCP, so its bottom is ``size - grasp_depth``
    below the TCP.
    """
    return support_height + control_cfg.place_clearance + scene_cfg.cube_size - control_cfg.grasp_depth


def expert_actions(
    spec: TaskSpec,
    cube_poses: dict[str, tuple[float, float, float, float]],
    bin_positions: dict[str, tuple[float, float]],
    scene_cfg,
    control_cfg,
    rng: np.random.Generator,
    bin_jitter: float = 0.02,
) -> tuple[Action, Action]:
    """Ground-truth pick and place for ``spec``.

    Args:
        cube_poses: colour -> (x, y, z_centre, yaw) of every cube, read from the simulator.
        bin_positions: colour -> (x, y) of every bin.
        bin_jitter: the place point is drawn uniformly within +-bin_jitter of the bin centre (varied labels).
    """
    if spec.obj not in cube_poses:
        raise KeyError(f"No {spec.obj} cube in the scene.")
    cx, cy, cz, cyaw = cube_poses[spec.obj]
    top = cz + scene_cfg.cube_size / 2
    pick = Action(cx, cy, cyaw % (math.pi / 2), top, tcp_height_pick(top, control_cfg))

    if spec.task == PICK_PLACE:
        if spec.dst not in bin_positions:
            raise KeyError(f"No {spec.dst} bin in the scene.")
        bx, by = bin_positions[spec.dst]
        dx, dy = rng.uniform(-bin_jitter, bin_jitter, size=2)
        support = scene_cfg.bin_floor_thickness
        place = Action(bx + float(dx), by + float(dy), 0.0, support, tcp_height_place(support, scene_cfg, control_cfg))
    elif spec.task == STACK:
        if spec.dst not in cube_poses:
            raise KeyError(f"No {spec.dst} cube in the scene.")
        tx, ty, tz, tyaw = cube_poses[spec.dst]
        support = tz + scene_cfg.cube_size / 2
        place = Action(tx, ty, tyaw % (math.pi / 2), support, tcp_height_place(support, scene_cfg, control_cfg))
    else:
        raise ValueError(f"Unknown task '{spec.task}'.")
    return pick, place
