"""Camera processing: overhead RGB-D image -> top-down orthographic colour and height maps (pure PyTorch).

Each pixel of the depth image is back-projected with the camera intrinsics, transformed to the world frame with
the camera pose (ROS optical convention: +x right, +y down, +z forward), and dropped into the workspace grid.
Every grid cell keeps the highest point that falls into it (what a top-down view would see) and the colour of
that point. Cells that receive no point (rare: the camera resolution is finer than the grid) are filled from
their neighbours.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vlamanip.geometry import WorkspaceGrid, quat_to_matrix


def depth_to_world_points(
    depth: torch.Tensor, intrinsics: torch.Tensor, cam_pos: torch.Tensor, cam_quat_ros: torch.Tensor
) -> torch.Tensor:
    """Back-project depth images to world points.

    Args:
        depth: (N, H, W) distance to the image plane (metres); non-finite or <= 0 values are invalid.
        intrinsics: (N, 3, 3) pinhole matrices.
        cam_pos: (N, 3) camera position in the world.
        cam_quat_ros: (N, 4) camera orientation (w, x, y, z) in the ROS optical convention.

    Returns:
        (N, H * W, 3) world points; invalid pixels are NaN.
    """
    n, h, w = depth.shape
    dev = depth.device
    v, u = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    pix = torch.stack((u.float() + 0.5, v.float() + 0.5, torch.ones_like(u, dtype=torch.float32)), dim=-1)
    rays = pix.reshape(1, -1, 3) @ torch.linalg.inv(intrinsics).transpose(1, 2)  # (N, HW, 3), z = 1
    d = depth.reshape(n, -1, 1).float()
    d = torch.where(torch.isfinite(d) & (d > 0), d, torch.full_like(d, float("nan")))
    points_cam = rays * d
    rot = quat_to_matrix(cam_quat_ros.float())  # camera -> world
    return points_cam @ rot.transpose(1, 2) + cam_pos.float()[:, None, :]


def project_to_heightmap(
    points: torch.Tensor, colors: torch.Tensor, grid: WorkspaceGrid, table_z: float = 0.0, fill_iterations: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-down maps from world points.

    Args:
        points: (N, P, 3) world points (NaN = invalid).
        colors: (N, P, 3) colours in [0, 1].

    Returns:
        color_map (N, 3, H, W) in [0, 1] and height_map (N, 1, H, W) = height above the table (>= 0).
    """
    n = points.shape[0]
    hm, wm = grid.height, grid.width
    rc = grid.world_to_pixel(points[..., :2].nan_to_num(nan=-1e6))
    valid = grid.in_bounds(rc) & torch.isfinite(points).all(-1)
    cell = torch.where(valid, rc[..., 0] * wm + rc[..., 1], torch.zeros_like(rc[..., 0]))
    z = torch.where(valid, points[..., 2] - table_z, torch.full_like(points[..., 2], -1e3))

    height = torch.full((n, hm * wm), -1e3, device=points.device)
    height.scatter_reduce_(1, cell, z, reduce="amax", include_self=True)
    # colour of the top-most point(s) of each cell (mean over points within 2 mm of the top)
    top = valid & (z >= height.gather(1, cell) - 0.002)
    weight = top.float()
    color_sum = torch.zeros(n, hm * wm, 3, device=points.device)
    color_sum.scatter_add_(1, cell[..., None].expand(-1, -1, 3), colors.float() * weight[..., None])
    count = torch.zeros(n, hm * wm, device=points.device).scatter_add_(1, cell, weight)

    filled = count > 0
    color = color_sum / count.clamp(min=1)[..., None]
    height = torch.where(filled, height, torch.zeros_like(height)).clamp(min=0.0)

    color = color.reshape(n, hm, wm, 3).permute(0, 3, 1, 2)
    height = height.reshape(n, 1, hm, wm)
    filled = filled.reshape(n, 1, hm, wm).float()
    for _ in range(fill_iterations):  # fill empty cells with the mean of their filled neighbours
        if bool(filled.all()):
            break
        kernel = torch.ones(1, 1, 3, 3, device=points.device)
        nb_count = F.conv2d(filled, kernel, padding=1)
        nb_color = F.conv2d((color * filled).reshape(-1, 1, hm, wm), kernel, padding=1).reshape(n, 3, hm, wm)
        nb_height = F.conv2d(height * filled, kernel, padding=1)
        newly = (filled == 0) & (nb_count > 0)
        color = torch.where(newly, nb_color / nb_count.clamp(min=1), color)
        height = torch.where(newly, nb_height / nb_count.clamp(min=1), height)
        filled = torch.maximum(filled, newly.float())
    return color, height


def rgbd_to_heightmap(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat_ros: torch.Tensor,
    grid: WorkspaceGrid,
    table_z: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rgb`` (N, H, W, 3) uint8 or float in [0, 1]; ``depth`` (N, H, W) or (N, H, W, 1)."""
    if depth.dim() == 4:
        depth = depth[..., 0]
    rgb = rgb[..., :3].float()
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    points = depth_to_world_points(depth, intrinsics, cam_pos, cam_quat_ros)
    return project_to_heightmap(points, rgb.reshape(rgb.shape[0], -1, 3), grid, table_z)
