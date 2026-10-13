"""Plot training and validation curves from one or more runs' metrics.jsonl.

    python scripts/plot_training.py runs/full runs/full_v2 --out docs/training_curves.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(run_dir: str) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            (val if "val_both" in row else train).append(row)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="docs/training_curves.png")
    args = parser.parse_args()

    fig, (ax_loss, ax_val) = plt.subplots(1, 2, figsize=(11, 4))
    for run in args.runs:
        name = os.path.basename(os.path.normpath(run))
        train, val = load(run)
        ax_loss.plot([r["it"] for r in train], [r["loss"] for r in train], label=f"{name} train", lw=1)
        vit = [r["it"] for r in val]
        ax_loss.plot(vit, [r["val_loss_pick"] + r["val_loss_place"] for r in val], "o--", ms=3, label=f"{name} val")
        ax_val.plot(vit, [100 * r["val_both"] for r in val], "o-", ms=3, label=name)

    ax_loss.set(xlabel="iteration", ylabel="loss (pick + place cross-entropy)", title="Loss")
    ax_val.set(xlabel="iteration", ylabel="%", title="Validation: pick on object and place on target", ylim=(0, 102))
    for ax in (ax_loss, ax_val):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
