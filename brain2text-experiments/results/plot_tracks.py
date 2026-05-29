"""
results/plot_tracks.py
----------------------
Generate WER comparison plots for each track.

Metrics per EXPERIMENT_DESIGN.md:
  1. WER at epoch 10   — how fast the model learns
  2. Slope = (WER_ep2 − WER_ep20) / 18  — learning efficiency
  3. Best WER across all epochs

Usage:
    python results/plot_tracks.py --track B --out results/track_B.png
    python results/plot_tracks.py --all   --out_dir results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from leaderboard import list_runs

BASELINE_WER = 0.3673
TARGET_WER   = 0.10
TRACK_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868",
                "D": "#C44E52", "E": "#8172B2"}


def _compute_slope(wer_history: list[float]) -> float | None:
    if len(wer_history) < 2:
        return None
    return (wer_history[0] - wer_history[-1]) / max(len(wer_history) - 1, 1)


def plot_track(track: str, out_path: str, profile: str = "toy"):
    runs = [r for r in list_runs(profile=profile) if r["expt_id"].startswith(track)]
    if not runs:
        print(f"No runs found for track {track}/{profile}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Track {track} — {profile} profile", fontsize=14, fontweight="bold")

    expts    = [r["expt_id"] for r in runs]
    wer10s   = [r.get("wer_at_ep10") or float("nan") for r in runs]
    slopes   = [r.get("slope")      or float("nan") for r in runs]
    best_wers = [r.get("best_wer")  or float("nan") for r in runs]
    color    = TRACK_COLORS.get(track, "#999999")

    x    = np.arange(len(expts))
    bar_kw = dict(color=color, alpha=0.8, edgecolor="white")

    # ── WER @ epoch 10 ──
    ax = axes[0]
    ax.bar(x, wer10s, **bar_kw)
    ax.axhline(BASELINE_WER, ls="--", color="gray", lw=1, label=f"Baseline {BASELINE_WER}")
    ax.axhline(TARGET_WER,   ls=":",  color="green", lw=1, label=f"Target {TARGET_WER}")
    ax.set_title("WER @ epoch 10 (lower=better)")
    ax.set_xticks(x); ax.set_xticklabels(expts, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=8)

    # ── Slope ──
    ax = axes[1]
    ax.bar(x, slopes, **bar_kw)
    ax.axhline(0, ls="-", color="black", lw=0.5)
    ax.set_title("Learning slope (higher=better)")
    ax.set_xticks(x); ax.set_xticklabels(expts, rotation=30, ha="right")

    # ── Best WER ──
    ax = axes[2]
    ax.bar(x, best_wers, **bar_kw)
    ax.axhline(BASELINE_WER, ls="--", color="gray", lw=1)
    ax.axhline(TARGET_WER,   ls=":",  color="green", lw=1)
    ax.set_title("Best WER (lower=better)")
    ax.set_xticks(x); ax.set_xticklabels(expts, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track",   default=None, choices=list("ABCDE"),
                        help="Single track to plot")
    parser.add_argument("--all",     action="store_true", help="Plot all tracks")
    parser.add_argument("--out",     default="results/track.png")
    parser.add_argument("--out_dir", default="results/", help="Used with --all")
    parser.add_argument("--profile", default="toy",  choices=["toy", "full"])
    args = parser.parse_args()

    if args.all:
        for t in "ABCDE":
            plot_track(t, f"{args.out_dir}/track_{t}_{args.profile}.png", args.profile)
    elif args.track:
        plot_track(args.track, args.out, args.profile)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
