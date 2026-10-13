"""Workspace grid, pixel <-> world conversion, yaw bins and quaternion helpers (pure PyTorch).

Top-down map convention (same orientation as the overhead camera image, robot at the bottom):

    row r  <->  x = x_max - (r + 0.5) * res      (row 0 = far edge of the table)
    col c  <->  y = y_max - (c + 0.5) * res      (col 0 = robot's left, +y)

Gripper yaw is the rotation about world +z of a top-down grasp. Cubes and bins look the same every 90 deg, so
yaw is represented modulo 90 deg in ``num_rotations`` bins: bin k <-> k * 90 / num_rotations degrees.
Quaternions are (w, x, y, z), as in Isaac Lab.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WorkspaceGrid:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    resolution: float
    num_rotations: int = 8

    @classmethod
    def from_config(cls, ws_cfg) -> "WorkspaceGrid":
        return cls(tuple(ws_cfg.x_range), tuple(ws_cfg.y_range), float(ws_cfg.resolution), int(ws_cfg.num_rotations))

    @property
    def height(self) -> int:  # rows
        return int(round((self.x_range[1] - self.x_range[0]) / self.resolution))

    @property
    def width(self) -> int:  # cols
        return int(round((self.y_range[1] - self.y_range[0]) / self.resolution))

    @property
    def rotation_step(self) -> float:
        """Radians per yaw bin."""
        return (math.pi / 2) / self.num_rotations

    # ------------------------------------------------------------------ positions
    def world_to_pixel(self, xy: torch.Tensor) -> torch.Tensor:
        """(..., 2) world x, y -> (..., 2) integer (row, col); may be outside the grid (see ``in_bounds``)."""
        row = torch.floor((self.x_range[1] - xy[..., 0]) / self.resolution)
        col = torch.floor((self.y_range[1] - xy[..., 1]) / self.resolution)
        return torch.stack((row, col), dim=-1).long()

    def pixel_to_world(self, rc: torch.Tensor) -> torch.Tensor:
        """(..., 2) (row, col) -> (..., 2) world x, y of the pixel centre."""
        rc = rc.float()
        x = self.x_range[1] - (rc[..., 0] + 0.5) * self.resolution
        y = self.y_range[1] - (rc[..., 1] + 0.5) * self.resolution
        return torch.stack((x, y), dim=-1)

    def in_bounds(self, rc: torch.Tensor) -> torch.Tensor:
        return (rc[..., 0] >= 0) & (rc[..., 0] < self.height) & (rc[..., 1] >= 0) & (rc[..., 1] < self.width)

    # ---------------------------------------------------------------- rotations
    def yaw_to_bin(self, yaw: torch.Tensor) -> torch.Tensor:
        """Yaw (rad, any value) -> nearest bin index in [0, num_rotations)."""
        k = torch.round(wrap_quarter(yaw) / self.rotation_step).long()
        return torch.remainder(k, self.num_rotations)

    def bin_to_yaw(self, k: torch.Tensor) -> torch.Tensor:
        """Bin index -> yaw in [0, pi/2)."""
        return k.float() * self.rotation_step

    # ------------------------------------------------------------- flat indices
    def flat_index(self, rot: torch.Tensor, rc: torch.Tensor) -> torch.Tensor:
        """Index into a (num_rotations, H, W) map flattened to one dimension."""
        return (rot * self.height + rc[..., 0]) * self.width + rc[..., 1]

    def unflatten_index(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inverse of ``flat_index``: returns (rot, (row, col))."""
        rot = index // (self.height * self.width)
        rem = index % (self.height * self.width)
        return rot, torch.stack((rem // self.width, rem % self.width), dim=-1)


def wrap_quarter(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw modulo 90 deg, in [0, pi/2)."""
    return torch.remainder(yaw, math.pi / 2)


def yaw_distance_quarter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Smallest absolute difference between two yaws modulo 90 deg, in [0, pi/4]."""
    d = torch.remainder(a - b, math.pi / 2)
    return torch.minimum(d, math.pi / 2 - d)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Rotation about +z by ``yaw`` -> (..., 4) quaternion (w, x, y, z)."""
    half = 0.5 * yaw
    zeros = torch.zeros_like(yaw)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation matrix -> (..., 4) quaternion (w, x, y, z) with w >= 0."""
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    candidates = torch.stack(
        (
            torch.stack((1 + trace, m[..., 2, 1] - m[..., 1, 2], m[..., 0, 2] - m[..., 2, 0], m[..., 1, 0] - m[..., 0, 1]), -1),
            torch.stack((m[..., 2, 1] - m[..., 1, 2], 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], m[..., 0, 1] + m[..., 1, 0], m[..., 0, 2] + m[..., 2, 0]), -1),
            torch.stack((m[..., 0, 2] - m[..., 2, 0], m[..., 0, 1] + m[..., 1, 0], 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], m[..., 1, 2] + m[..., 2, 1]), -1),
            torch.stack((m[..., 1, 0] - m[..., 0, 1], m[..., 0, 2] + m[..., 2, 0], m[..., 1, 2] + m[..., 2, 1], 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]), -1),
        ),
        dim=-2,
    )  # (..., 4, 4): row i is 4*q*q_i for the branch with the largest diagonal term
    best = torch.stack((trace, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]), -1).argmax(-1)
    q = torch.gather(candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)).squeeze(-2)
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def look_at_quat_ros(eye: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Camera orientation (ROS optical convention: +x right, +y down, +z forward) looking from eye to target."""
    forward = target - eye
    forward = forward / torch.linalg.norm(forward, dim=-1, keepdim=True)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=eye.dtype, device=eye.device).expand_as(forward)
    right = torch.linalg.cross(forward, up)
    right = right / torch.linalg.norm(right, dim=-1, keepdim=True)
    down = torch.linalg.cross(forward, right)
    return quat_from_matrix(torch.stack((right, down, forward), dim=-1))


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Yaw (rotation about +z) of an orientation, assuming it is close to upright."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# Hand pointing straight down (hand z axis = world -z): 180 deg about world x.
QUAT_HAND_DOWN = (0.0, 1.0, 0.0, 0.0)


def top_down_grasp_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Orientation of the Franka hand for a top-down grasp rotated by ``yaw`` about world +z."""
    down = torch.tensor(QUAT_HAND_DOWN, dtype=yaw.dtype, device=yaw.device).expand(*yaw.shape, 4)
    return quat_mul(quat_from_yaw(yaw), down)
