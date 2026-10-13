"""Execute one pick-and-place per environment and let the scene settle (shared by the expert, data and evaluation)."""

from __future__ import annotations

import torch


def execute(scene, executor, pick: torch.Tensor, place: torch.Tensor, settle_s: float, max_time_s: float = 30.0,
            recorder=None, preview_s: float = 0.5) -> float:
    """Run ``pick``/``place`` ((N, 4) = x, y, yaw, z_tcp) in every environment until all are done, then settle.

    With a recorder, a frame is captured every second control step (25 fps at 50 Hz). Returns the wall time (s).
    """
    import time

    executor.start(torch.arange(scene.num_envs, device=scene.device), pick, place)
    if recorder is not None:
        for _ in range(int(preview_s * 25)):  # show the targets before the arm moves
            recorder.record()
    t0 = time.time()
    max_steps = int(max_time_s / executor.control_dt)
    settle_steps = int(settle_s / executor.control_dt)
    k = 0
    while k < max_steps and not executor.all_done():
        frame = recorder is not None and k % 2 == 0
        executor.step(render=frame)
        if frame:
            recorder.record()
        k += 1
    for j in range(settle_steps):
        frame = recorder is not None and j % 2 == 0
        executor.step(render=frame)
        if frame:
            recorder.record()
    return time.time() - t0
