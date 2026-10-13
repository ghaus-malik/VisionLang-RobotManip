"""Generate imitation-learning demonstrations with the scripted expert.

Each demo is one random layout (4 cubes + 4 bins) and one instruction (pick-and-place or stacking). Stored:
    * the overhead RGB-D image with its calibration and the top-down colour / height maps built from it,
    * the instruction text, task type, object and destination colours,
    * the expert pick and place poses (metric) and the same targets as (rotation bin, row, col) map cells,
    * with ``--execute``: the outcome of running the expert in simulation (grasp, placement, success, time).

The expert reads ground-truth poses; that is allowed here because this is how the training labels are made.
Batches of ``--num_envs`` layouts are simulated in parallel. Videos of ``--videos`` randomly chosen demos are
recorded when executing.

    CUDA_VISIBLE_DEVICES=0 python scripts/generate_demos.py --headless --num_demos 200 --execute --videos 6 \
        --out data/demos_pilot.h5
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--num_demos", type=int, default=200)
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=None, help="default: seeds.demos_train from the config")
parser.add_argument("--split", default="train", choices=("train", "heldout"), help="instruction template set")
parser.add_argument("--stack_fraction", type=float, default=0.5)
parser.add_argument("--execute", action="store_true", help="run the expert and store the outcome")
parser.add_argument("--videos", type=int, default=0, help="record this many randomly chosen demos (needs --execute)")
parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "demos.h5"))
parser.add_argument("--video_dir", default=os.path.join(PROJECT_ROOT, "videos", "demos"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.videos and not args.execute:
    parser.error("--videos needs --execute")
simulation_app = AppLauncher(args).app

import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.control.executor import PickPlaceExecutor  # noqa: E402
from vlamanip.data.store import DemoWriter  # noqa: E402
from vlamanip.evaluation.success import judge_episode  # noqa: E402
from vlamanip.expert.expert import expert_actions  # noqa: E402
from vlamanip.geometry import WorkspaceGrid  # noqa: E402
from vlamanip.language.templates import PICK_PLACE, STACK, sample_task  # noqa: E402
from vlamanip.perception.heightmap import rgbd_to_heightmap  # noqa: E402
from vlamanip.sim.layouts import sample_layout  # noqa: E402
from vlamanip.sim.rollout import execute  # noqa: E402
from vlamanip.sim.scene import TabletopScene  # noqa: E402
from vlamanip.viz.recorder import EpisodeRecorder  # noqa: E402


def main():
    cfg = load_config()
    seed = cfg.seeds.demos_train if args.seed is None else args.seed
    grid = WorkspaceGrid.from_config(cfg.workspace)
    colors = list(cfg.scene.colors)
    n_env = min(args.num_envs, args.num_demos)
    n_batches = -(-args.num_demos // n_env)
    video_ids = set(np.random.default_rng(seed + 1).choice(args.num_demos, size=min(args.videos, args.num_demos), replace=False).tolist())

    scene = TabletopScene(cfg, num_envs=n_env, video=bool(video_ids))
    ex = PickPlaceExecutor(scene, cfg)
    writer = DemoWriter(args.out, cfg, attrs={"seed": seed, "split": args.split, "executed": args.execute})
    stats = {"demos": 0, "success": 0, "grasp": 0, "placed": 0, "time": [], "per_task": {PICK_PLACE: [0, 0], STACK: [0, 0]}}
    t_start = time.time()

    for b in range(n_batches):
        # every batch has its own seeded generators, so any batch can be regenerated on its own
        np_rng = np.random.default_rng([seed, b])
        text_rng = random.Random(seed * 100003 + b)
        layouts = [sample_layout(np_rng, cfg.scene, cfg.layout, colors, colors) for _ in range(n_env)]
        tasks = [STACK if np_rng.random() < args.stack_fraction else PICK_PLACE for _ in range(n_env)]
        specs = [sample_task(colors, text_rng, args.split, t) for t in tasks]

        scene.reset(layouts, ready_joint_pos=cfg.control.ready_joint_pos)
        scene.settle(0.5)
        obs = scene.capture_overhead()
        color, height = rgbd_to_heightmap(obs["rgb"], obs["depth"], obs["intrinsics"], obs["cam_pos"], obs["cam_quat_ros"], grid)
        start = [scene.cube_poses(i) for i in range(n_env)]
        bins = [scene.bin_positions(i) for i in range(n_env)]

        pick, place, actions = torch.zeros(n_env, 4), torch.zeros(n_env, 4), []
        for i in range(n_env):
            a_pick, a_place = expert_actions(specs[i], start[i], bins[i], cfg.scene, cfg.control, np_rng)
            pick[i] = torch.tensor([a_pick.x, a_pick.y, a_pick.yaw, a_pick.z_tcp])
            place[i] = torch.tensor([a_place.x, a_place.y, a_place.yaw, a_place.z_tcp])
            actions.append({"pick": a_pick.to_dict(), "place": a_place.to_dict()})

        outcomes = [None] * n_env
        if args.execute:
            ids = [i for i in range(n_env) if b * n_env + i in video_ids and b * n_env + i < args.num_demos]
            recorder = None
            if ids:
                paths = [os.path.join(args.video_dir, f"demo_{b * n_env + i:05d}_{specs[i].task}.mp4") for i in ids]
                recorder = EpisodeRecorder(scene, ex, ids, paths, [specs[i].text for i in ids], [actions[i] for i in ids])
            execute(scene, ex, pick, place, cfg.success.settle_s, recorder=recorder)
            for i in range(n_env):
                outcomes[i] = judge_episode(specs[i], start[i], scene.cube_poses(i), bins[i], bool(ex.grasp_ok[i]), cfg.scene, cfg.success)
                outcomes[i]["execution_time_s"] = round(ex.elapsed_s[i].item(), 2)
                outcomes[i]["waypoint_timeouts"] = int(ex.timeouts[i])
            if recorder is not None:
                for p in recorder.finish([outcomes[i]["success"] for i in ids]):
                    print(f"  video: {p}", flush=True)

        keep = min(n_env, args.num_demos - b * n_env)
        for i in range(keep):
            writer.add(
                rgb=obs["rgb"][i], depth=obs["depth"][i], intrinsics=obs["intrinsics"][i], cam_pos=obs["cam_pos"][i],
                cam_quat_ros=obs["cam_quat_ros"][i], color_map=color[i], height_map=height[i, 0],
                spec=specs[i], actions=actions[i], grid=grid,
                meta={"layout": layouts[i].to_dict(), "start_poses": start[i], "bins": bins[i], "batch": b, "index_in_batch": i},
                outcome=outcomes[i],
            )
            stats["demos"] += 1
            if outcomes[i] is not None:
                o = outcomes[i]
                stats["success"] += o["success"]
                stats["grasp"] += o["grasp_ok"]
                stats["placed"] += o["placed"]
                stats["time"].append(o["execution_time_s"])
                stats["per_task"][specs[i].task][0] += o["success"]
                stats["per_task"][specs[i].task][1] += 1
        elapsed = time.time() - t_start
        msg = f"[DEMOS] batch {b + 1}/{n_batches}: {stats['demos']} demos, {elapsed:.0f} s"
        if args.execute:
            msg += f", expert success {stats['success']}/{stats['demos']}"
        print(msg, flush=True)

    summary = {
        "demos": stats["demos"],
        "seed": seed,
        "split": args.split,
        "wall_time_s": round(time.time() - t_start, 1),
        "file": args.out,
    }
    if args.execute:
        summary.update({
            "expert_success_rate": stats["success"] / stats["demos"],
            "grasp_success": stats["grasp"] / stats["demos"],
            "placement_success": stats["placed"] / stats["demos"],
            "mean_execution_time_s": float(np.mean(stats["time"])),
            "per_task_success": {k: (v[0] / v[1] if v[1] else None) for k, v in stats["per_task"].items()},
            "per_task_count": {k: v[1] for k, v in stats["per_task"].items()},
        })
    writer.close(summary)
    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[DEMOS] done: {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
