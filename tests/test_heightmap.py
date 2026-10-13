import math

import torch

from vlamanip.perception.heightmap import rgbd_to_heightmap
from vlamanip.perception.reference import map_error_mask, reference_heightmap

# ROS optical frame looking straight down: image x -> world -y, image y -> world -x, optical z -> world -z
CAM_QUAT_DOWN = torch.tensor([[0.0, 1 / math.sqrt(2), -1 / math.sqrt(2), 0.0]])


def synthetic_overhead(cfg, cube_xy=(0.5, 0.1), cube_size=0.045):
    """Exact depth/colour image of a table with one axis-aligned cube, seen by the overhead camera."""
    c = cfg.camera
    w, h = c.width, c.height
    fx = w * c.focal_length / c.horizontal_aperture
    K = torch.tensor([[[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]]])
    cam = torch.tensor([c.overhead_pos])
    v, u = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
    # ray direction in camera frame (z = 1) -> world: x_w = cam_x - y_c * d, y_w = cam_y - x_c * d
    xc, yc = (u - w / 2) / fx, (v - h / 2) / fx
    d_table = cam[0, 2].item()
    xw, yw = cam[0, 0] - yc * d_table, cam[0, 1] - xc * d_table
    on_cube = (abs(xw - cube_xy[0]) < cube_size / 2) & (abs(yw - cube_xy[1]) < cube_size / 2)
    # rays that hit the top face of the cube (re-test at the top height for exactness)
    d_top = d_table - cube_size
    xt, yt = cam[0, 0] - yc * d_top, cam[0, 1] - xc * d_top
    hit_top = (abs(xt - cube_xy[0]) < cube_size / 2) & (abs(yt - cube_xy[1]) < cube_size / 2)
    depth = torch.where(hit_top, torch.tensor(d_top), torch.tensor(d_table))[None]
    rgb = torch.full((1, h, w, 3), 0.5)
    rgb[0][hit_top] = torch.tensor([0.1, 0.2, 0.9])
    return rgb, depth, K, cam, on_cube


def test_cube_appears_at_its_position_and_height(cfg, grid):
    rgb, depth, K, cam, _ = synthetic_overhead(cfg)
    color, height = rgbd_to_heightmap(rgb, depth, K, cam, CAM_QUAT_DOWN, grid)
    assert color.shape == (1, 3, 160, 160) and height.shape == (1, 1, 160, 160)
    r, c = grid.world_to_pixel(torch.tensor([0.5, 0.1])).tolist()
    assert abs(height[0, 0, r, c].item() - 0.045) < 1e-3
    assert torch.allclose(color[0, :, r, c], torch.tensor([0.1, 0.2, 0.9]), atol=1e-3)
    # the table around it is flat and grey
    r2, c2 = grid.world_to_pixel(torch.tensor([0.5, -0.1])).tolist()
    assert height[0, 0, r2, c2].item() < 1e-3
    assert torch.allclose(color[0, :, r2, c2], torch.tensor([0.5, 0.5, 0.5]), atol=1e-3)


def test_cube_footprint_area(cfg, grid):
    rgb, depth, K, cam, _ = synthetic_overhead(cfg)
    _, height = rgbd_to_heightmap(rgb, depth, K, cam, CAM_QUAT_DOWN, grid)
    cells = (height > 0.02).sum().item()
    expected = (0.045 / grid.resolution) ** 2
    assert abs(cells - expected) / expected < 0.2


def test_camera_map_matches_reference_raster(cfg, grid):
    rgb, depth, K, cam, _ = synthetic_overhead(cfg)
    _, height = rgbd_to_heightmap(rgb, depth, K, cam, CAM_QUAT_DOWN, grid)
    ref = reference_heightmap({"blue": (0.5, 0.1, 0.0225, 0.0)}, {}, cfg.scene, grid)
    assert not map_error_mask(height[0, 0], ref).any()


def test_reference_raster_bins_and_rotated_cube(cfg, grid):
    s = cfg.scene
    ref = reference_heightmap({"red": (0.4, 0.15, s.cube_size / 2, math.pi / 4)}, {"blue": (0.6, -0.1)}, s, grid)
    at = lambda x, y: ref[tuple(grid.world_to_pixel(torch.tensor([x, y])).tolist())].item()  # noqa: E731
    assert abs(at(0.6, -0.1) - s.bin_floor_thickness) < 1e-6
    assert abs(at(0.6 + s.bin_inner / 2 + s.bin_wall_thickness / 2, -0.1) - s.bin_wall_height) < 1e-6
    assert abs(at(0.4, 0.15) - s.cube_size) < 1e-6
    # a 45-degree cube reaches farther along x than an axis-aligned one, but not its corner region along x=y
    assert abs(at(0.4 + 0.029, 0.15) - s.cube_size) < 1e-6
    assert at(0.4 + 0.02, 0.15 + 0.02) == 0.0


def test_error_mask_flags_occluder(cfg, grid):
    ref = reference_heightmap({}, {}, cfg.scene, grid)
    height = ref.clone()
    height[10:20, 10:20] = 0.3  # something hovering over the table
    wrong = map_error_mask(height, ref)
    assert wrong.sum().item() == 100


def test_world_to_image_inverts_back_projection(cfg):
    from vlamanip.viz.video import world_to_image

    rgb, depth, K, cam, _ = synthetic_overhead(cfg)
    uv = world_to_image(torch.tensor([[0.6, 0.1, 0.0], [cam[0, 0].item(), cam[0, 1].item(), 0.0]]), K[0], cam[0], CAM_QUAT_DOWN[0])
    # the point straight below the camera is at the image centre; +y world is left, +x world is up
    assert torch.allclose(uv[1], torch.tensor([cfg.camera.width / 2, cfg.camera.height / 2]), atol=1e-3)
    assert uv[0, 0] < cfg.camera.width / 2 and uv[0, 1] < cfg.camera.height / 2


def test_no_holes_and_invalid_depth_ignored(cfg, grid):
    rgb, depth, K, cam, _ = synthetic_overhead(cfg)
    depth[0, :5, :] = float("inf")
    color, height = rgbd_to_heightmap(rgb, depth, K, cam, CAM_QUAT_DOWN, grid)
    assert torch.isfinite(height).all() and torch.isfinite(color).all()
    assert (height >= 0).all()
