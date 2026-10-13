"""Run the scripted expert (ground-truth poses) through the IK executor and measure how reliable it is.

    --stage 1   one cube + one bin at the nominal fixed positions (``--jitter`` adds a small seeded perturbation)
    --stage 2   four cubes + four bins, random layouts, pick-and-place instructions
    --stage 3   four cubes + four bins, random layouts, stacking instructions

All trials run in parallel (one environment each).

    CUDA_VISIBLE_DEVICES=0 python scripts/run_expert.py --headless --stage 1 --trials 20 --videos 4
Writes results/expert/stage{N}/summary.json and videos/expert/stage{N}/*.mp4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=1)
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--videos", type=int, default=4, help="record the first N trials")
parser.add_argument("--jitter", type=float, default=0.01, help="stage 1 only: cube position perturbation (m), yaw +-15 deg")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_time_s", type=float, default=30.0)
parser.add_argument("--out", default=None)
parser.add_argument("--video_dir", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import math  # noqa: E402
import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.control.executor import DONE, PickPlaceExecutor  # noqa: E402
from vlamanip.evaluation.success import judge_episode  # noqa: E402
from vlamanip.expert.expert import expert_actions  # noqa: E402
from vlamanip.language.templates import PICK_PLACE, STACK, make_instruction, sample_task  # noqa: E402
from vlamanip.sim.layouts import Layout, fixed_layout, sample_layout  # noqa: E402
from vlamanip.sim.rollout import execute  # noqa: E402
from vlamanip.sim.scene import TabletopScene  # noqa: E402
from vlamanip.viz.recorder import EpisodeRecorder  # noqa: E402


def make_trials(cfg, n: int, np_rng: np.random.Generator, text_rng: random.Random):
    """Layouts and instructions for ``n`` trials of the selected stage."""
    if args.stage == 1:
        base = fixed_layout(1)
        obj, dst = next(iter(base.cubes)), next(iter(base.bins))
        layouts = []
        for _ in range(n):
            x, y, yaw = base.cubes[obj]
            dx, dy = np_rng.uniform(-args.jitter, args.jitter, size=2)
            dyaw = np_rng.uniform(-math.radians(15), math.radians(15)) if args.jitter > 0 else 0.0
            layouts.append(Layout(cubes={obj: (x + dx, y + dy, yaw + dyaw)}, bins=dict(base.bins)))
        specs = [make_instruction(PICK_PLACE, obj, dst, text_rng, "train") for _ in range(n)]
        return layouts, specs, [obj], [dst]
    colors = list(cfg.scene.colors)
    layouts = [sample_layout(np_rng, cfg.scene, cfg.layout, colors, colors) for _ in range(n)]
    task = PICK_PLACE if args.stage == 2 else STACK
    specs = [sample_task(colors, text_rng, "train", task) for _ in range(n)]
    return layouts, specs, colors, colors


def main():
    cfg = load_config()
    np_rng, text_rng = np.random.default_rng(args.seed), random.Random(args.seed)
    out = args.out or os.path.join(PROJECT_ROOT, "results", "expert", f"stage{args.stage}")
    video_dir = args.video_dir or os.path.join(PROJECT_ROOT, "videos", "expert", f"stage{args.stage}")
    os.makedirs(out, exist_ok=True)
    n = args.trials
    layouts, specs, cube_colors, bin_colors = make_trials(cfg, n, np_rng, text_rng)

    scene = TabletopScene(cfg, num_envs=n, cube_colors=cube_colors, bin_colors=bin_colors, video=args.videos > 0)
    ex = PickPlaceExecutor(scene, cfg)
    scene.reset(layouts, ready_joint_pos=cfg.control.ready_joint_pos)
    scene.settle(0.5)
    start = [scene.cube_poses(i) for i in range(n)]
    bins = [scene.bin_positions(i) for i in range(n)]

    pick, place, actions = torch.zeros(n, 4), torch.zeros(n, 4), []
    for i in range(n):
        a_pick, a_place = expert_actions(specs[i], start[i], bins[i], cfg.scene, cfg.control, np_rng)
        pick[i] = torch.tensor([a_pick.x, a_pick.y, a_pick.yaw, a_pick.z_tcp])
        place[i] = torch.tensor([a_place.x, a_place.y, a_place.yaw, a_place.z_tcp])
        actions.append({"pick": a_pick.to_dict(), "place": a_place.to_dict()})

    n_vid = min(args.videos, n)
    recorder = None
    if n_vid:
        recorder = EpisodeRecorder(
            scene, ex, list(range(n_vid)), [os.path.join(video_dir, f"trial_{i:02d}.mp4") for i in range(n_vid)],
            [s.text for s in specs[:n_vid]], actions[:n_vid],
        )
    wall = execute(scene, ex, pick, place, cfg.success.settle_s, args.max_time_s, recorder)

    trials = []
    for i in range(n):
        outcome = judge_episode(specs[i], start[i], scene.cube_poses(i), bins[i], bool(ex.grasp_ok[i]), cfg.scene, cfg.success)
        trials.append({
            "instruction": specs[i].text, "task": specs[i].task, "object": specs[i].obj, "destination": specs[i].dst,
            "layout": layouts[i].to_dict(), "expert": actions[i], **outcome,
            "finished": bool(ex.phase[i] == DONE),
            "waypoint_timeouts": int(ex.timeouts[i]),
            "execution_time_s": round(ex.elapsed_s[i].item(), 2),
        })
    if recorder is not None:
        recorder.finish([t["success"] for t in trials[:n_vid]])

    summary = {
        "stage": args.stage,
        "trials": n,
        "grasp_success": sum(t["grasp_ok"] for t in trials) / n,
        "placement_success": sum(t["placed"] for t in trials) / n,
        "success_rate": sum(t["success"] for t in trials) / n,
        "disturbed_trials": sum(bool(t["disturbed"]) for t in trials),
        "mean_execution_time_s": float(np.mean([t["execution_time_s"] for t in trials])),
        "total_waypoint_timeouts": sum(t["waypoint_timeouts"] for t in trials),
        "wall_time_s": round(wall, 1),
        "seed": args.seed,
        "jitter_m": args.jitter if args.stage == 1 else None,
        "per_trial": trials,
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[EXPERT stage {args.stage}] success {summary['success_rate']:.0%}  grasp {summary['grasp_success']:.0%}  "
        f"placed {summary['placement_success']:.0%}  disturbed {summary['disturbed_trials']}  "
        f"mean time {summary['mean_execution_time_s']:.1f} s  timeouts {summary['total_waypoint_timeouts']}  wall {wall:.1f} s",
        flush=True,
    )
    for i, t in enumerate(trials):
        if not t["success"]:
            print(f"  FAILED trial {i:02d}: {t['instruction']!r} grasp={t['grasp_ok']} placed={t['placed']} "
                  f"disturbed={t['disturbed']} timeouts={t['waypoint_timeouts']} cube={t['final_cube_pos']}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
