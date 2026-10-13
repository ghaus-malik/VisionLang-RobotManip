"""Run the learned policy on one natural-language instruction and save a video.

    CUDA_VISIBLE_DEVICES=0 python scripts/run_instruction.py --headless \
        --instruction "Put the red cube in the blue bin." --layout_seed 7

The scene is a random layout (4 cubes + 4 bins) drawn from ``--layout_seed``. The policy sees only the overhead
camera and the instruction. If the instruction names a cube colour and a destination colour, the outcome is
scored with ground truth and shown as SUCCESS / FAILURE in the video.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--instruction", required=True)
parser.add_argument("--layout_seed", type=int, default=0)
parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "runs", "full_v2", "last.pt"))
parser.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "videos", "instructions"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.control.executor import PickPlaceExecutor  # noqa: E402
from vlamanip.evaluation.success import judge_episode  # noqa: E402
from vlamanip.geometry import WorkspaceGrid  # noqa: E402
from vlamanip.language.templates import parse_instruction  # noqa: E402
from vlamanip.perception.heightmap import rgbd_to_heightmap  # noqa: E402
from vlamanip.policy import Policy  # noqa: E402
from vlamanip.sim.layouts import sample_layout  # noqa: E402
from vlamanip.sim.rollout import execute  # noqa: E402
from vlamanip.sim.scene import TabletopScene  # noqa: E402
from vlamanip.viz.recorder import EpisodeRecorder  # noqa: E402


def main():
    cfg = load_config()
    grid = WorkspaceGrid.from_config(cfg.workspace)
    colors = list(cfg.scene.colors)
    layout = sample_layout(np.random.default_rng(args.layout_seed), cfg.scene, cfg.layout, colors, colors)
    policy = Policy.load(args.checkpoint, cfg)
    scene = TabletopScene(cfg, num_envs=1, video=True)
    ex = PickPlaceExecutor(scene, cfg)
    scene.reset([layout], ready_joint_pos=cfg.control.ready_joint_pos)
    scene.settle(0.5)

    obs = scene.capture_overhead()
    color, height = rgbd_to_heightmap(obs["rgb"], obs["depth"], obs["intrinsics"], obs["cam_pos"], obs["cam_quat_ros"], grid)
    pred = policy.predict(color, height, [args.instruction])[0]
    print(f"[RUN] instruction: {args.instruction!r}")
    for k in ("pick", "place"):
        p = pred[k]
        print(f"[RUN] {k}: x={p['x']:.3f} y={p['y']:.3f} yaw={np.degrees(p['yaw']):.1f} deg z_tcp={p['z_tcp']:.3f}")

    spec = parse_instruction(args.instruction, colors)
    start, bins = scene.cube_poses(0), scene.bin_positions(0)
    slug = re.sub(r"[^a-z0-9]+", "_", args.instruction.lower()).strip("_")[:60]
    path = os.path.join(args.out_dir, f"{slug}_seed{args.layout_seed}.mp4")
    recorder = EpisodeRecorder(scene, ex, [0], [path], [args.instruction], [pred])
    pick = torch.tensor([[pred["pick"]["x"], pred["pick"]["y"], pred["pick"]["yaw"], pred["pick"]["z_tcp"]]])
    place = torch.tensor([[pred["place"]["x"], pred["place"]["y"], pred["place"]["yaw"], pred["place"]["z_tcp"]]])
    execute(scene, ex, pick, place, cfg.success.settle_s, recorder=recorder)

    result = None
    if spec is not None:
        result = judge_episode(spec, start, scene.cube_poses(0), bins, bool(ex.grasp_ok[0]), cfg.scene, cfg.success)
        result["execution_time_s"] = round(ex.elapsed_s[0].item(), 2)
        final = recorder.finish([result["success"]])[0]
    else:
        for _ in range(25):
            recorder.record()
        for w in recorder.writers:
            w.close()
        final = path
    print(f"[RUN] outcome: {json.dumps(result)}")
    print(f"[RUN] video: {final}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
