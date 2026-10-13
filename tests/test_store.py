import json

import numpy as np
import pytest
import torch

h5py = pytest.importorskip("h5py")

from vlamanip.data.store import DemoWriter, action_cell, read_strings  # noqa: E402
from vlamanip.language.templates import TaskSpec  # noqa: E402


def _action(x, y, yaw=0.3):
    return {"x": x, "y": y, "yaw": yaw, "z_tcp": 0.02, "support_height": 0.045}


def test_action_cell_matches_grid(grid):
    rot, r, c = action_cell(_action(0.5, 0.1), grid)
    assert (r, c) == tuple(grid.world_to_pixel(torch.tensor([0.5, 0.1])).tolist())
    assert rot == int(grid.yaw_to_bin(torch.tensor(0.3)))
    with pytest.raises(ValueError):
        action_cell(_action(2.0, 0.0), grid)


def test_writer_round_trip(tmp_path, cfg, grid):
    path = str(tmp_path / "demos.h5")
    w = DemoWriter(path, cfg, attrs={"seed": 0})
    spec = TaskSpec("stack", "red", "blue", "Stack the red cube on the blue cube.")
    for k in range(3):
        w.add(
            rgb=torch.full((256, 256, 4), k, dtype=torch.uint8), depth=torch.ones(256, 256), intrinsics=torch.eye(3),
            cam_pos=torch.zeros(3), cam_quat_ros=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            color_map=torch.full((3, grid.height, grid.width), 0.5), height_map=torch.zeros(grid.height, grid.width),
            spec=spec, actions={"pick": _action(0.4, 0.1), "place": _action(0.6, -0.1, 0.0)}, grid=grid,
            meta={"batch": 0, "index_in_batch": k}, outcome={"success": True} if k else None,
        )
    w.close({"demos": 3})
    with h5py.File(path) as f:
        assert f["rgb"].shape == (3, 256, 256, 3) and f["rgb"][2, 0, 0, 0] == 2
        assert f["color_map"].dtype == np.uint8 and f["color_map"][0, 0, 0, 0] == 128
        assert f["pick_cell"].shape == (3, 3)
        assert f["outcome"][:].tolist() == [-1, 1, 1]
        assert read_strings(f["instruction"])[1] == spec.text
        assert json.loads(read_strings(f["meta"])[2])["index_in_batch"] == 2
        assert f.attrs["num_demos"] == 3
