"""Train the language-conditioned pick-and-place policy by imitation of the scripted expert's demos.

    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --data data/demos_train.h5 --run full
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py --data data/demos_pilot.h5 --run overfit --overfit 200 \
        --set train.iterations=1500 train.val_every=250

Logs to runs/<run>/ (TensorBoard events, metrics.jsonl, checkpoints, the resolved config).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch  # noqa: E402

from vlamanip.config import load_config  # noqa: E402
from vlamanip.data.augment import augment  # noqa: E402
from vlamanip.data.dataset import DemoDataset  # noqa: E402
from vlamanip.geometry import WorkspaceGrid  # noqa: E402
from vlamanip.models.clip_encoder import FrozenCLIP  # noqa: E402
from vlamanip.models.losses import argmax_cells, soft_cross_entropy, soft_targets  # noqa: E402
from vlamanip.policy import build_net, prepare_maps  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(PROJECT_ROOT, "data", "demos_train.h5"))
    p.add_argument("--run", default="full")
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "runs"))
    p.add_argument("--overfit", type=int, default=0, help="train and validate on the first N demos, no augmentation")
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=3e-4")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def lr_at(it: int, cfg) -> float:
    """Linear warm-up then cosine decay to 5% of the peak."""
    t = cfg.train
    if it < t.warmup_iterations:
        return t.lr * (it + 1) / t.warmup_iterations
    progress = (it - t.warmup_iterations) / max(1, t.iterations - t.warmup_iterations)
    return t.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))


def main():
    args = parse_args()
    cfg = load_config(overrides=args.set)
    t, m = cfg.train, cfg.model
    torch.manual_seed(t.seed)
    torch.backends.cudnn.benchmark = True  # fixed input sizes: let cuDNN pick the fastest convolution kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = args.device
    run_dir = os.path.join(args.out, args.run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "args": vars(args)}, f, indent=2)

    grid = WorkspaceGrid.from_config(cfg.workspace)
    data = DemoDataset(args.data, grid, cfg.scene, dev, limit=args.overfit or None)
    if args.overfit:
        train_idx = val_idx = torch.arange(data.n, device=dev)
    else:
        train_idx, val_idx = data.split(t.val_fraction, t.seed)
    print(f"[TRAIN] {data.n} demos ({len(train_idx)} train / {len(val_idx)} val), {len(data.unique_texts)} distinct instructions", flush=True)

    clip = FrozenCLIP(m.clip_name, m.clip_image_size).to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        words, word_mask, sentence = [], [], []
        for i in range(0, len(data.unique_texts), 256):
            w, k, s = clip.encode_text(data.unique_texts[i:i + 256])
            words.append(w.float()), word_mask.append(k), sentence.append(s.float())
        words, word_mask, sentence = torch.cat(words), torch.cat(word_mask), torch.cat(sentence)

    net = build_net(m, clip, grid.num_rotations, grid.height).to(dev).to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[TRAIN] policy network: {n_params / 1e6:.1f} M trainable parameters (CLIP frozen)", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=t.lr, weight_decay=t.weight_decay)

    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(run_dir)
    log_file = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    gen = torch.Generator(device=dev).manual_seed(t.seed)
    aug = not args.overfit

    def forward(idx, train: bool):
        maps = prepare_maps(data.color[idx], data.height[idx], m.height_scale)
        labels = [data.pick_cell[idx], data.place_cell[idx]]
        if train and aug:
            maps, labels = augment(maps, labels, grid.num_rotations, t.augment_rot90, t.augment_flip, t.augment_shift_px, gen)
        maps = maps.contiguous(memory_format=torch.channels_last)
        ti = data.text_index[idx]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=t.amp):
            clip_img = clip.encode_image((maps[:, :3] + 1) / 2)
            pick, place = net(maps, clip_img, words[ti], word_mask[ti], sentence[ti])
        tp = soft_targets(labels[0], grid.num_rotations, grid.height, grid.width, m.pick_sigma_px, m.rot_sigma_bins)
        tq = soft_targets(labels[1], grid.num_rotations, grid.height, grid.width, m.place_sigma_px, m.rot_sigma_bins)
        loss_pick, loss_place = soft_cross_entropy(pick, tp), soft_cross_entropy(place, tq)
        return loss_pick, loss_place, pick, place

    @torch.no_grad()
    def validate():
        net.eval()
        totals, lp, lq, n = {}, 0.0, 0.0, 0
        for i in range(0, len(val_idx), 64):
            idx = val_idx[i:i + 64]
            a, b, pick, place = forward(idx, train=False)
            lp, lq, n = lp + a.item() * len(idx), lq + b.item() * len(idx), n + len(idx)
            scores = data.score(idx, argmax_cells(pick.float()), argmax_cells(place.float()))
            for k, v in scores.items():
                totals[k] = totals.get(k, 0) + v.sum().item()
                if k in ("pick_on_object", "place_on_target", "both"):
                    for task, mask in (("stack", data.is_stack[idx]), ("pick_place", ~data.is_stack[idx])):
                        key = f"{k}/{task}"
                        totals[key] = totals.get(key, 0) + v[mask].sum().item()
        net.train()
        n_stack = data.is_stack[val_idx].sum().item()
        out = {"val_loss_pick": lp / n, "val_loss_place": lq / n}
        for k, v in totals.items():
            denom = n_stack if k.endswith("/stack") else (n - n_stack if k.endswith("/pick_place") else n)
            out[f"val_{k}"] = v / max(1, denom)
        return out

    def save(name, it, metrics):
        torch.save({"net": net.state_dict(), "iteration": it, "metrics": metrics, "config": cfg.to_dict()}, os.path.join(run_dir, name))

    best, t0, t_log = -1.0, time.time(), time.time()
    net.train()
    for it in range(t.iterations):
        for g in opt.param_groups:
            g["lr"] = lr_at(it, cfg)
        idx = train_idx[torch.randint(0, len(train_idx), (t.batch_size,), device=dev, generator=gen)]
        loss_pick, loss_place, _, _ = forward(idx, train=True)
        loss = loss_pick + loss_place
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
        opt.step()

        if (it + 1) % t.log_every == 0:
            rate = t.log_every / (time.time() - t_log)
            t_log = time.time()
            rec = {"it": it + 1, "loss": loss.item(), "loss_pick": loss_pick.item(), "loss_place": loss_place.item(),
                   "lr": opt.param_groups[0]["lr"], "grad_norm": grad_norm.item(), "it_per_s": rate}
            for k, v in rec.items():
                if k != "it":
                    tb.add_scalar(f"train/{k}", v, it + 1)
            log_file.write(json.dumps(rec) + "\n")
            log_file.flush()
            eta = (t.iterations - it - 1) / max(rate, 1e-6) / 60
            print(f"[TRAIN] it {it + 1:6d}  loss {loss.item():.3f} (pick {loss_pick.item():.3f}, place {loss_place.item():.3f})"
                  f"  lr {rec['lr']:.2e}  {rate:.1f} it/s  eta {eta:.1f} min", flush=True)

        if (it + 1) % t.val_every == 0 or it + 1 == t.iterations:
            metrics = validate()
            metrics["it"] = it + 1
            for k, v in metrics.items():
                if k != "it":
                    tb.add_scalar(k.replace("val_", "val/"), v, it + 1)
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()
            print("[VAL] it {it}: pick on object {p:.1%}  place on target {q:.1%}  both {b:.1%}  pick yaw {y:.1%}  "
                  "(stack both {bs:.1%}, pick-place both {bp:.1%})".format(
                      it=it + 1, p=metrics["val_pick_on_object"], q=metrics["val_place_on_target"], b=metrics["val_both"],
                      y=metrics["val_pick_yaw"], bs=metrics["val_both/stack"], bp=metrics["val_both/pick_place"]), flush=True)
            if metrics["val_both"] >= best:  # ties go to the later (longer-trained) checkpoint
                best = metrics["val_both"]
                save("best.pt", it + 1, metrics)
        if (it + 1) % t.save_every == 0:
            save("last.pt", it + 1, {})
    save("last.pt", t.iterations, {})
    print(f"[TRAIN] done in {(time.time() - t0) / 60:.1f} min, best val both {best:.1%}", flush=True)
    tb.close()
    log_file.close()


if __name__ == "__main__":
    main()
