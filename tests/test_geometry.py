import math

import pytest
import torch

from vlamanip.config import load_config
from vlamanip.geometry import (
    quat_to_matrix,
    top_down_grasp_quat,
    yaw_distance_quarter,
    yaw_from_quat,
)


def test_grid_is_160_by_160(grid):
    assert (grid.height, grid.width) == (160, 160)


def test_pixel_world_round_trip(grid):
    rc = torch.tensor([[0, 0], [159, 159], [80, 37]])
    assert torch.equal(grid.world_to_pixel(grid.pixel_to_world(rc)), rc)


def test_orientation_matches_camera_image(grid):
    # far edge of the workspace (large x) is row 0; the robot's left (+y) is column 0
    far_left = grid.world_to_pixel(torch.tensor([0.77, 0.27]))
    near_right = grid.world_to_pixel(torch.tensor([0.23, -0.27]))
    assert far_left.tolist() == [2, 2]
    assert near_right.tolist() == [157, 157]


def test_in_bounds(grid):
    rc = grid.world_to_pixel(torch.tensor([[0.5, 0.0], [0.9, 0.0], [0.5, -0.3]]))
    assert grid.in_bounds(rc).tolist() == [True, False, False]


@pytest.mark.parametrize("deg, k", [(0, 0), (11.25, 1), (45, 4), (88, 0), (90, 0), (100, 1), (-11.25, 7)])
def test_yaw_bins_are_modulo_90(grid, deg, k):
    assert grid.yaw_to_bin(torch.tensor(math.radians(deg))).item() == k


def test_flat_index_round_trip(grid):
    rot = torch.tensor([0, 3, 7])
    rc = torch.tensor([[0, 0], [12, 140], [159, 159]])
    r2, rc2 = grid.unflatten_index(grid.flat_index(rot, rc))
    assert torch.equal(r2, rot) and torch.equal(rc2, rc)


def test_yaw_distance_quarter():
    d = yaw_distance_quarter(torch.tensor(math.radians(5)), torch.tensor(math.radians(85)))
    assert d.item() == pytest.approx(math.radians(10), abs=1e-6)


def test_top_down_grasp_points_down_and_keeps_yaw():
    yaw = torch.tensor([0.0, 0.4, 1.2])
    rot = quat_to_matrix(top_down_grasp_quat(yaw))
    assert torch.allclose(rot[:, :, 2], torch.tensor([0.0, 0.0, -1.0]).expand(3, 3), atol=1e-6)
    # the hand x axis turns with the yaw (up to the 180 deg flip about x)
    hand_x = rot[:, :2, 0]
    assert torch.allclose(torch.atan2(hand_x[:, 1], hand_x[:, 0]), yaw, atol=1e-6)


def test_yaw_from_quat():
    from vlamanip.geometry import quat_from_yaw

    yaw = torch.tensor([-1.0, 0.0, 0.7, 3.0])
    assert torch.allclose(yaw_from_quat(quat_from_yaw(yaw)), yaw, atol=1e-6)


def test_quat_from_matrix_round_trip():
    from vlamanip.geometry import quat_from_matrix

    q = torch.nn.functional.normalize(torch.randn(100, 4), dim=-1)
    q = torch.where(q[:, :1] < 0, -q, q)
    assert torch.allclose(quat_from_matrix(quat_to_matrix(q)), q, atol=1e-5)


def test_look_at_points_optical_axis_at_target():
    from vlamanip.geometry import look_at_quat_ros

    eye, target = torch.tensor([1.3, 0.85, 0.75]), torch.tensor([0.45, 0.0, 0.05])
    rot = quat_to_matrix(look_at_quat_ros(eye, target))
    forward = (target - eye) / torch.linalg.norm(target - eye)
    assert torch.allclose(rot[:, 2], forward, atol=1e-6)
    assert rot[2, 1] < 0  # image "down" points down in the world


def test_config_override():
    cfg = load_config(overrides=["train.batch_size=4", "workspace.x_range=[0.2, 0.8]"])
    assert cfg.train.batch_size == 4 and cfg.workspace.x_range == [0.2, 0.8]
    with pytest.raises(KeyError):
        load_config(overrides=["train.no_such_key=1"])
