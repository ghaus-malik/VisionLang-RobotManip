"""Scene smoke test: spawn the tabletop, capture overhead RGB-D, build the top-down map and check it.

Checks (ground truth vs. what the camera pipeline reports):
  * every cube shows up at its true position with the height of its top face and its colour,
  * every bin floor and wall shows up with the right height,
  * the whole map agrees with a ground-truth height raster (this also measures how much the arm hides),
  * several candidate observation poses are compared and the one that hides the least is reported.

    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_scene.py --headless
Outputs images, the raw observation (obs.pt) and a JSON summary to results/smoke_scene/.
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
parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results", "smoke_scene"))
parser.add_argument("--layout_seed", type=int, default=None, help="random layout seed (default: fixed layout)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import time  # noqa: E402

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.geometry import WorkspaceGrid  # noqa: E402
from vlamanip.perception.heightmap import rgbd_to_heightmap  # noqa: E402
from vlamanip.perception.reference import map_error_mask, reference_heightmap  # noqa: E402
from vlamanip.sim.layouts import fixed_layout, sample_layout  # noqa: E402
from vlamanip.sim.scene import COLOR_RGB, TabletopScene  # noqa: E402

CANDIDATE_POSES = {
    "isaac_default": [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741],
    "folded": [0.0, -1.2, 0.0, -2.95, 0.0, 1.75, 0.785],
    "folded_high": [0.0, -1.76, 0.0, -3.0, 0.0, 1.3, 0.785],
    "side": [-1.57, -0.8, 0.0, -2.6, 0.0, 1.8, 0.785],
}


def to_png(x: torch.Tensor) -> np.ndarray:
    return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def classify_color(rgb: torch.Tensor) -> str:
    """Nearest object colour by chroma (brightness removed), so lighting and shading do not matter."""
    chroma = lambda v: torch.nn.functional.normalize(v - v.mean(), dim=0)  # noqa: E731
    return max(COLOR_RGB, key=lambda name: float(chroma(rgb) @ chroma(torch.tensor(COLOR_RGB[name]))))


def main():
    cfg = load_config()
    grid = WorkspaceGrid.from_config(cfg.workspace)
    os.makedirs(args.out, exist_ok=True)
    layout = fixed_layout(2) if args.layout_seed is None else sample_layout(
        np.random.default_rng(args.layout_seed), cfg.scene, cfg.layout, cfg.scene.colors, cfg.scene.colors
    )
    scene = TabletopScene(cfg, num_envs=1, video=True)
    results = {"layout": layout.to_dict(), "checks": {}, "poses": {}}

    def observe(joints):
        scene.reset([layout], ready_joint_pos=joints)
        scene.settle(0.5)
        obs = scene.capture_overhead()
        color, height = rgbd_to_heightmap(
            obs["rgb"], obs["depth"], obs["intrinsics"], obs["cam_pos"], obs["cam_quat_ros"], grid
        )
        ref = reference_heightmap(scene.cube_poses(0), scene.bin_positions(0), cfg.scene, grid).to(height.device)
        wrong = map_error_mask(height[0, 0], ref)
        return obs, color, height, ref, wrong

    # ------------------------------------------------ observation pose candidates
    for name, joints in CANDIDATE_POSES.items():
        obs, _, _, _, wrong = observe(joints)
        links = scene.robot.data.body_pos_w[0] - scene.origins[0]
        hand = links[scene.robot.find_bodies("panda_hand")[0][0]]
        results["poses"][name] = {
            "wrong_cells": int(wrong.sum()),
            "hand_pos": [round(v, 3) for v in hand.tolist()],
            "max_link_x": round(links[:, 0].max().item(), 3),
        }
        imageio.imwrite(os.path.join(args.out, f"pose_{name}_rgb.png"), obs["rgb"][0].cpu().numpy())
        print(f"[POSE] {name}: {results['poses'][name]}", flush=True)

    best = min(results["poses"], key=lambda k: results["poses"][k]["wrong_cells"])
    results["observation_pose"] = best
    print(f"[POSE] least occluding pose: {best}", flush=True)

    # ---------------------------------------------------------- map vs. truth
    obs, color, height, ref, wrong = observe(CANDIDATE_POSES[best])
    torch.save({k: v.cpu() for k, v in obs.items()} | {"reference": ref.cpu()}, os.path.join(args.out, "obs.pt"))
    hm = height[0, 0]

    def sample(x, y, r=1):
        rc = grid.world_to_pixel(torch.tensor([x, y]))
        return hm[rc[0] - r: rc[0] + r + 1, rc[1] - r: rc[1] + r + 1].median().item(), color[0, :, rc[0], rc[1]]

    checks = results["checks"]
    for c, (x, y, z, yaw) in scene.cube_poses(0).items():
        h, rgb = sample(x, y)
        top = z + cfg.scene.cube_size / 2
        ok_h = abs(h - top) < 0.005
        ok_c = classify_color(rgb.cpu()) == c
        checks[f"cube_{c}"] = {"true_top": round(top, 4), "map_height": round(h, 4), "rgb": [round(v, 2) for v in rgb.tolist()], "pass": bool(ok_h and ok_c)}
    for c, (x, y) in scene.bin_positions(0).items():
        h_floor, _ = sample(x, y, r=2)
        h_wall, _ = sample(x + cfg.scene.bin_inner / 2 + cfg.scene.bin_wall_thickness / 2, y, r=0)
        ok = abs(h_floor - cfg.scene.bin_floor_thickness) < 0.005 and abs(h_wall - cfg.scene.bin_wall_height) < 0.008
        checks[f"bin_{c}"] = {"floor": round(h_floor, 4), "wall": round(h_wall, 4), "pass": bool(ok)}
    frac = wrong.float().mean().item()
    checks["map_vs_reference"] = {"wrong_fraction": round(frac, 4), "pass": frac < 0.01}
    K = obs["intrinsics"][0]
    fx_expected = cfg.camera.width * cfg.camera.focal_length / cfg.camera.horizontal_aperture
    checks["intrinsics"] = {"fx": round(K[0, 0].item(), 2), "expected": round(fx_expected, 2), "pass": bool(abs(K[0, 0] - fx_expected) < 1.0)}
    results["camera"] = {"pos": obs["cam_pos"][0].tolist(), "quat_ros": obs["cam_quat_ros"][0].tolist()}

    imageio.imwrite(os.path.join(args.out, "overhead_rgb.png"), obs["rgb"][0].cpu().numpy())
    depth = obs["depth"][0]
    imageio.imwrite(os.path.join(args.out, "overhead_depth.png"), ((depth - depth.min()) / (depth.max() - depth.min()) * 255).byte().cpu().numpy())
    imageio.imwrite(os.path.join(args.out, "map_color.png"), to_png(color[0]))
    imageio.imwrite(os.path.join(args.out, "map_height.png"), (hm / 0.06).clamp(0, 1).mul(255).byte().cpu().numpy())
    imageio.imwrite(os.path.join(args.out, "map_reference.png"), (ref / 0.06).clamp(0, 1).mul(255).byte().cpu().numpy())
    imageio.imwrite(os.path.join(args.out, "map_wrong.png"), wrong.byte().mul(255).cpu().numpy())
    top, side = scene.capture_video_frames()
    imageio.imwrite(os.path.join(args.out, "video_top.png"), top[0].cpu().numpy())
    imageio.imwrite(os.path.join(args.out, "video_side.png"), side[0].cpu().numpy())

    t = time.time()
    for _ in range(200):
        scene.step()
    results["physics_steps_per_s"] = round(200 / (time.time() - t), 1)

    passed = sum(v["pass"] for v in checks.values())
    results["passed"] = f"{passed}/{len(checks)}"
    for k, v in checks.items():
        print(f"[{'PASS' if v['pass'] else 'FAIL'}] {k}: {v}")
    print(f"{passed}/{len(checks)} checks passed; physics {results['physics_steps_per_s']} steps/s", flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
    # Kit can hang on shutdown when cameras were used; results are already on disk.
    sys.stdout.flush()
    os._exit(0)
