import math

import torch

from vlamanip.control.waypoints import angle_error, gripper_yaw, step_angle_toward, step_toward


def test_gripper_yaw_range_and_equivalence():
    yaw = torch.linspace(-7.0, 7.0, 1001)
    g = gripper_yaw(yaw)
    assert (g >= -math.pi / 4 - 1e-6).all() and (g < math.pi / 4 + 1e-6).all()
    # differs from the input by a multiple of 90 degrees
    k = (yaw - g) / (math.pi / 2)
    assert torch.allclose(k, k.round(), atol=1e-4)


def test_step_toward_bounded_and_exact_when_close():
    cur = torch.zeros(2, 3)
    goal = torch.tensor([[1.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    out = step_toward(cur, goal, 0.01)
    assert torch.allclose(out[0], torch.tensor([0.01, 0.0, 0.0]))
    assert torch.allclose(out[1], goal[1])


def test_step_angle_takes_short_way():
    cur = torch.tensor([3.0])
    goal = torch.tensor([-3.0])  # shortest path crosses +pi
    out = step_angle_toward(cur, goal, 0.1)
    assert out.item() > 3.0
    assert angle_error(torch.tensor([3.1]), torch.tensor([-3.1])).item() < 0.1
