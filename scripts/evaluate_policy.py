"""Evaluate the learned policy in simulation on unseen layouts (evaluation seeds never used for training).

For every trial: random layout (4 cubes + 4 bins) -> overhead RGB-D -> top-down maps -> policy(maps, instruction)
-> pick/place poses -> IK executor -> ground-truth scoring. The policy sees only the camera maps and the text;
ground-truth poses are read only to score the outcome.

Suites:
    pick_place           pick-and-place, instruction phrasings seen in training
    stack                stacking, phrasings seen in training
    pick_place_heldout   pick-and-place, phrasings never seen in training
    stack_heldout        stacking, phrasings never seen in training
    *_reordered          stress test: unseen phrasings that name the destination before the object

Metrics (per suite and overall): correct object selection, grasp success, correct destination selection,
bin placement success, stacking success, overall success, average execution time.

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_policy.py --headless --checkpoint runs/full_v2/last.pt
Writes results/eval/summary.json + summary.md and videos/{pick_place,stacking}/{success,failure}/*.mp4.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

SUITES = {
    "pick_place": ("pick_place", "train"),
    "stack": ("stack", "train"),
    "pick_place_heldout": ("pick_place", "heldout"),
    "stack_heldout": ("stack", "heldout"),
    "pick_place_reordered": ("pick_place", "reordered"),
    "stack_reordered": ("stack", "reordered"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "runs", "full_v2", "last.pt"))
parser.add_argument("--suites", nargs="*", default=list(SUITES), choices=list(SUITES))
parser.add_argument("--trials", type=int, default=100, help="trials per suite")
parser.add_argument("--num_envs", type=int, default=50)
parser.add_argument("--seed", type=int, default=None, help="default: seeds.eval from the config")
parser.add_argument("--record", type=int, default=6, help="envs recorded per batch")
parser.add_argument("--keep_videos", type=int, default=3, help="videos kept per suite and outcome")
parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "eval"))
parser.add_argument("--video_root", default=os.path.join(PROJECT_ROOT, "videos"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.control.executor import PickPlaceExecutor  # noqa: E402
from vlamanip.evaluation.success import judge_episode, point_in_bin, point_on_cube  # noqa: E402
from vlamanip.geometry import WorkspaceGrid  # noqa: E402
from vlamanip.language.templates import sample_task  # noqa: E402
from vlamanip.perception.heightmap import rgbd_to_heightmap  # noqa: E402
from vlamanip.policy import Policy  # noqa: E402
from vlamanip.sim.layouts import sample_layout  # noqa: E402
from vlamanip.sim.rollout import execute  # noqa: E402
from vlamanip.sim.scene import TabletopScene  # noqa: E402
from vlamanip.viz.recorder import EpisodeRecorder  # noqa: E402

LIFTED_MOVE = 0.02  # the instructed cube moved at least this far -> it was the one grasped


def score_trial(spec, pred, start, final, bins, grasp_ok, cfg) -> dict:
    """All metrics of one trial from the policy's prediction and ground-truth poses before / after."""
    s = cfg.scene
    obj_xy = start[spec.obj][:2]
    pick_xy = (pred["pick"]["x"], pred["pick"]["y"])
    place_xy = (pred["place"]["x"], pred["place"]["y"])
    correct_object = point_on_cube(pick_xy, obj_xy, s)
    picked_other = next((c for c in start if c != spec.obj and point_on_cube(pick_xy, start[c][:2], s)), None)
    if spec.task == "pick_place":
        correct_destination = point_in_bin(place_xy, bins[spec.dst], s)
    else:
        correct_destination = point_on_cube(place_xy, start[spec.dst][:2], s)
    outcome = judge_episode(spec, start, final, bins, grasp_ok, s, cfg.success)
    moved = math.dist(final[spec.obj][:2], obj_xy) > LIFTED_MOVE or final[spec.obj][2] - start[spec.obj][2] > LIFTED_MOVE
    return {
        "correct_object": bool(correct_object),
        "picked_other_cube": picked_other,
        "grasp_success": bool(grasp_ok and correct_object and moved),
        "correct_destination": bool(correct_destination),
        **outcome,
    }


def aggregate(trials: list[dict]) -> dict:
    n = len(trials)
    if n == 0:
        return {}
    mean = lambda k: sum(bool(t[k]) for t in trials) / n  # noqa: E731
    out = {
        "trials": n,
        "correct_object_selection": mean("correct_object"),
        "grasp_success": mean("grasp_success"),
        "correct_destination_selection": mean("correct_destination"),
        "overall_success": mean("success"),
        "mean_execution_time_s": float(np.mean([t["execution_time_s"] for t in trials])),
        "mean_inference_time_ms": float(np.mean([t["inference_ms"] for t in trials])),
    }
    pp = [t for t in trials if t["task"] == "pick_place"]
    st = [t for t in trials if t["task"] == "stack"]
    if pp:
        out["bin_placement_success"] = sum(t["placed"] for t in pp) / len(pp)
    if st:
        out["stacking_success"] = sum(t["placed"] for t in st) / len(st)
    return out


def main():
    cfg = load_config()
    seed = cfg.seeds.eval if args.seed is None else args.seed
    grid = WorkspaceGrid.from_config(cfg.workspace)
    colors = list(cfg.scene.colors)
    policy = Policy.load(args.checkpoint, cfg)
    n_env = min(args.num_envs, args.trials)
    scene = TabletopScene(cfg, num_envs=n_env, video=args.record > 0)
    ex = PickPlaceExecutor(scene, cfg)
    os.makedirs(args.out, exist_ok=True)
    tmp_video = os.path.join(args.video_root, "_tmp_eval")
    all_trials, report = [], {"checkpoint": args.checkpoint, "seed": seed, "suites": {}}
    t_start = time.time()

    for suite in args.suites:
        j = list(SUITES).index(suite)  # fixed per suite, so running a subset reproduces the same trials
        task, split = SUITES[suite]
        video_dir = os.path.join(args.video_root, "pick_place" if task == "pick_place" else "stacking")
        kept = {"success": 0, "failure": 0}
        trials = []
        for b in range(-(-args.trials // n_env)):
            np_rng = np.random.default_rng([seed, j, b])
            text_rng = random.Random(seed * 1000 + j * 100 + b)
            layouts = [sample_layout(np_rng, cfg.scene, cfg.layout, colors, colors) for _ in range(n_env)]
            specs = [sample_task(colors, text_rng, split, task) for _ in range(n_env)]

            scene.reset(layouts, ready_joint_pos=cfg.control.ready_joint_pos)
            scene.settle(0.5)
            obs = scene.capture_overhead()
            color, height = rgbd_to_heightmap(obs["rgb"], obs["depth"], obs["intrinsics"], obs["cam_pos"], obs["cam_quat_ros"], grid)
            torch.cuda.synchronize()
            t_inf = time.time()
            preds = policy.predict(color, height, [s.text for s in specs])
            torch.cuda.synchronize()
            inference_ms = (time.time() - t_inf) * 1000 / n_env  # batched; per-instruction share
            start = [scene.cube_poses(i) for i in range(n_env)]
            bins = [scene.bin_positions(i) for i in range(n_env)]
            pick = torch.tensor([[p["pick"]["x"], p["pick"]["y"], p["pick"]["yaw"], p["pick"]["z_tcp"]] for p in preds])
            place = torch.tensor([[p["place"]["x"], p["place"]["y"], p["place"]["yaw"], p["place"]["z_tcp"]] for p in preds])

            rec_ids = list(range(min(args.record, n_env)))
            recorder = None
            if rec_ids:
                paths = [os.path.join(tmp_video, f"{suite}_{b * n_env + i:03d}.mp4") for i in rec_ids]
                recorder = EpisodeRecorder(scene, ex, rec_ids, paths, [specs[i].text for i in rec_ids], [preds[i] for i in rec_ids])
            execute(scene, ex, pick, place, cfg.success.settle_s, recorder=recorder)

            keep = min(n_env, args.trials - b * n_env)
            batch_trials = []
            for i in range(n_env):
                r = score_trial(specs[i], preds[i], start[i], scene.cube_poses(i), bins[i], bool(ex.grasp_ok[i]), cfg)
                r.update({
                    "suite": suite, "task": task, "split": split, "instruction": specs[i].text,
                    "object": specs[i].obj, "destination": specs[i].dst, "layout": layouts[i].to_dict(),
                    "prediction": preds[i], "execution_time_s": round(ex.elapsed_s[i].item(), 2),
                    "inference_ms": inference_ms, "waypoint_timeouts": int(ex.timeouts[i]),
                })
                batch_trials.append(r)
            if recorder is not None:
                files = recorder.finish([batch_trials[i]["success"] for i in rec_ids])
                for i, f in zip(rec_ids, files):
                    tag = "success" if batch_trials[i]["success"] else "failure"
                    if i < keep and kept[tag] < args.keep_videos:
                        os.makedirs(os.path.join(video_dir, tag), exist_ok=True)
                        dst = os.path.join(video_dir, tag, os.path.basename(f).replace(f"_{tag}.mp4", ".mp4"))
                        shutil.move(f, dst)
                        batch_trials[i]["video"] = os.path.relpath(dst, PROJECT_ROOT)
                        kept[tag] += 1
                    else:
                        os.remove(f)
            trials.extend(batch_trials[:keep])
            agg = aggregate(trials)
            print(f"[EVAL] {suite} {len(trials)}/{args.trials}: success {agg['overall_success']:.1%}  "
                  f"object {agg['correct_object_selection']:.1%}  grasp {agg['grasp_success']:.1%}  "
                  f"destination {agg['correct_destination_selection']:.1%}  ({time.time() - t_start:.0f} s)", flush=True)
        report["suites"][suite] = aggregate(trials)
        all_trials.extend(trials)

    report["overall"] = aggregate(all_trials)
    for name, keep_fn in (("seen_phrasings", lambda t: t["split"] == "train"),
                          ("heldout_phrasings", lambda t: t["split"] == "heldout"),
                          ("reordered_phrasings", lambda t: t["split"] == "reordered")):
        subset = [t for t in all_trials if keep_fn(t)]
        if subset:
            report[name] = aggregate(subset)
    report["wall_time_s"] = round(time.time() - t_start, 1)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "trials.jsonl"), "w") as f:
        for t in all_trials:
            f.write(json.dumps(t) + "\n")
    write_markdown(report, os.path.join(args.out, "summary.md"))
    shutil.rmtree(tmp_video, ignore_errors=True)
    print(open(os.path.join(args.out, "summary.md")).read(), flush=True)


def write_markdown(report: dict, path: str) -> None:
    cols = [("Object", "correct_object_selection"), ("Grasp", "grasp_success"), ("Destination", "correct_destination_selection"),
            ("Bin placement", "bin_placement_success"), ("Stacking", "stacking_success"), ("Overall", "overall_success")]
    lines = ["| Suite | Trials | " + " | ".join(c for c, _ in cols) + " | Exec. time (s) |",
             "|---|---|" + "---|" * len(cols) + "---|"]
    rows = list(report["suites"].items()) + [
        (k, report[k]) for k in ("seen_phrasings", "heldout_phrasings", "reordered_phrasings", "overall") if k in report
    ]
    for name, a in rows:
        cells = [f"{a[k]:.1%}" if k in a else "-" for _, k in cols]
        lines.append(f"| {name} | {a['trials']} | " + " | ".join(cells) + f" | {a['mean_execution_time_s']:.1f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
