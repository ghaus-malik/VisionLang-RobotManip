"""Ground-truth success checks (evaluation only; never used to choose actions)."""

from __future__ import annotations

import math


def cube_in_bin(cube_xyz, bin_xy, scene_cfg, success_cfg) -> bool:
    """Cube centre inside the bin interior and resting low (not on the rim)."""
    half = scene_cfg.bin_inner / 2 - success_cfg.bin_margin
    return (
        abs(cube_xyz[0] - bin_xy[0]) < half
        and abs(cube_xyz[1] - bin_xy[1]) < half
        and cube_xyz[2] < success_cfg.bin_max_height
    )


def cube_on_cube(top_xyz, bottom_xyz, scene_cfg, success_cfg) -> bool:
    """``top`` rests on ``bottom``: centred within the tolerance, one cube height higher."""
    dxy = math.hypot(top_xyz[0] - bottom_xyz[0], top_xyz[1] - bottom_xyz[1])
    dz = top_xyz[2] - bottom_xyz[2]
    return dxy < success_cfg.stack_xy_tolerance and abs(dz - scene_cfg.cube_size) < success_cfg.stack_z_tolerance


def judge_episode(spec, start_poses: dict, final_poses: dict, bin_positions: dict, grasp_ok: bool, scene_cfg, success_cfg) -> dict:
    """Outcome of one pick-and-place or stacking episode from ground-truth poses before and after.

    ``start_poses`` / ``final_poses``: colour -> (x, y, z, yaw); ``bin_positions``: colour -> (x, y).
    Success needs a verified grasp, the cube at the right destination and no other cube moved by more than
    ``success_cfg.disturb_tolerance``.
    """
    moved = final_poses[spec.obj]
    if spec.task == "pick_place":
        placed = cube_in_bin(moved[:3], bin_positions[spec.dst], scene_cfg, success_cfg)
    else:
        placed = cube_on_cube(moved[:3], final_poses[spec.dst][:3], scene_cfg, success_cfg)
    disturbed = [
        c for c in final_poses
        if c != spec.obj and math.dist(final_poses[c][:2], start_poses[c][:2]) > success_cfg.disturb_tolerance
    ]
    return {
        "grasp_ok": bool(grasp_ok),
        "placed": bool(placed),
        "disturbed": disturbed,
        "success": bool(grasp_ok and placed and not disturbed),
        "final_cube_pos": [round(v, 4) for v in moved[:3]],
    }


def point_in_bin(xy, bin_xy, scene_cfg) -> bool:
    """Is a predicted place point inside the bin footprint (interior)?"""
    half = scene_cfg.bin_inner / 2
    return abs(xy[0] - bin_xy[0]) < half and abs(xy[1] - bin_xy[1]) < half


def point_on_cube(xy, cube_xy, scene_cfg) -> bool:
    """Is a predicted point on the top face of a cube (within half the edge length of its centre)?"""
    return math.hypot(xy[0] - cube_xy[0], xy[1] - cube_xy[1]) < scene_cfg.cube_size / 2
