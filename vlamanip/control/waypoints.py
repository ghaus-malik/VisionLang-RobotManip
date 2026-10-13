"""Waypoint helpers for the top-down pick-and-place controller (pure PyTorch, batched over environments)."""

from __future__ import annotations

import math

import torch

QUARTER = math.pi / 2


def gripper_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Equivalent hand yaw in [-pi/4, pi/4) for a 90-degree-symmetric object.

    Cubes and bins look the same every 90 degrees, so the hand only ever needs to turn by at most 45 degrees from
    its neutral orientation; this keeps the last wrist joint far from its limits.
    """
    return torch.remainder(yaw + math.pi / 4, QUARTER) - math.pi / 4


def step_toward(current: torch.Tensor, goal: torch.Tensor, max_step: float) -> torch.Tensor:
    """Move each (N, D) point toward its goal by at most ``max_step`` (Euclidean)."""
    delta = goal - current
    dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(max_step / dist.clamp(min=1e-9), max=1.0)
    return current + delta * scale


def step_angle_toward(current: torch.Tensor, goal: torch.Tensor, max_step: float) -> torch.Tensor:
    """Move angles toward their goals along the shortest direction by at most ``max_step`` radians."""
    delta = torch.remainder(goal - current + math.pi, 2 * math.pi) - math.pi
    return current + delta.clamp(-max_step, max_step)


def angle_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Absolute shortest angular difference in [0, pi]."""
    return (torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi).abs()
