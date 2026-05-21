from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_NAMES = ["thumb", "index", "middle", "ring", "little"]


def _clean_results(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path, sep="\t")
    for col in ["mean_pearson", "rmse", "params", "peak_vram_mb", "train_seconds", "steps"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["status"].fillna("") == "ok"].copy()


def plot_metric_table(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    best = (
        df.sort_values("mean_pearson", ascending=False)
        .groupby("model", as_index=False)
        .first()
        .sort_values("mean_pearson", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.5 * len(best) + 1.5)))
    ax.axis("off")
    table_cols = ["model", "mean_pearson", "rmse", "params", "peak_vram_mb", "train_seconds"]
    table = ax.table(
        cellText=best[table_cols].round(4).astype(str).values,
        colLabels=table_cols,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(out_dir / "architecture_comparison_table.png", dpi=180)
    plt.close(fig)


def plot_progress(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in df.groupby("model"):
        group = group.sort_values("timestamp")
        ax.plot(range(1, len(group) + 1), group["mean_pearson"], marker="o", label=model)
    ax.set_title("Validation Mean Pearson by Run")
    ax.set_xlabel("Run index per model")
    ax.set_ylabel("Mean Pearson")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_pearson_progress.png", dpi=180)
    plt.close(fig)


def plot_vram_runtime(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = np.maximum(df["params"].fillna(0).to_numpy() / 2500, 20)
    for model, group in df.groupby("model"):
        ax.scatter(
            group["train_seconds"],
            group["mean_pearson"],
            s=np.maximum(group["params"].fillna(0).to_numpy() / 2500, 20),
            alpha=0.75,
            label=model,
        )
    ax.set_title("Performance vs Runtime")
    ax.set_xlabel("Train seconds")
    ax.set_ylabel("Mean Pearson")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_vs_performance.png", dpi=180)
    plt.close(fig)

    if df["peak_vram_mb"].fillna(0).max() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["peak_vram_mb"], df["mean_pearson"], s=sizes, alpha=0.75)
        for _, row in df.iterrows():
            ax.annotate(str(row["model"]), (row["peak_vram_mb"], row["mean_pearson"]), fontsize=8)
        ax.set_title("Performance vs Peak VRAM")
        ax.set_xlabel("Peak VRAM MB")
        ax.set_ylabel("Mean Pearson")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "vram_vs_performance.png", dpi=180)
        plt.close(fig)


def _latest_successful_run(df: pd.DataFrame, runs_dir: Path) -> Path | None:
    if df.empty:
        return None
    for run_id in reversed(df.sort_values("timestamp")["run_id"].tolist()):
        path = runs_dir / str(run_id)
        if path.exists():
            return path
    return None


def plot_history(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "history.json"
    if not path.exists():
        return
    history = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if history.empty:
        return
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss")
    ax1.plot(history["epoch"], history["val_loss"], label="val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE loss")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["val_mean_pearson"], color="tab:green", label="val pearson")
    ax2.set_ylabel("Mean Pearson")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "latest_loss_curves.png", dpi=180)
    plt.close(fig)


def plot_predictions(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "predictions.npz"
    if not path.exists():
        return
    data = np.load(path)
    pred = data["val_pred"]
    target = data["val_target"]
    n = min(len(pred), 300)

    corrs = []
    for i in range(target.shape[1]):
        p = pred[:, i] - pred[:, i].mean()
        t = target[:, i] - target[:, i].mean()
        corrs.append(float((p * t).sum() / max(np.sqrt((p**2).sum() * (t**2).sum()), 1e-12)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(TARGET_NAMES[: len(corrs)], corrs)
    ax.set_ylim(-1, 1)
    ax.set_title("Per-Finger Validation Pearson")
    ax.set_ylabel("Pearson r")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "per_finger_pearson.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(target.shape[1], 1, figsize=(11, 8), sharex=True)
    if target.shape[1] == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        ax.plot(target[:n, idx], label="actual", linewidth=1.5)
        ax.plot(pred[:n, idx], label="pred", linewidth=1.2, alpha=0.85)
        ax.set_ylabel(TARGET_NAMES[idx] if idx < len(TARGET_NAMES) else f"target {idx}")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Validation window")
    fig.suptitle("Predicted vs Actual Finger Trajectories")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_vs_actual_trajectories.png", dpi=180)
    plt.close(fig)


def plot_router(run_dir: Path, out_dir: Path) -> None:
    path = run_dir / "router.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    usage = payload.get("usage") or []
    entropy = payload.get("entropy") or []
    if usage:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([f"E{i}" for i in range(len(usage))], usage)
        ax.set_title("Expert Utilization")
        ax.set_ylabel("Mean router weight")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "expert_utilization.png", dpi=180)
        plt.close(fig)
    if entropy:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(entropy)
        ax.set_title("Router Entropy During Training")
        ax.set_xlabel("MoE training step")
        ax.set_ylabel("Entropy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "router_entropy.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECoG autoresearch plots.")
    parser.add_argument("--results-path", default="results.tsv")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)
    if not results_path.exists():
        raise FileNotFoundError(f"No results file found: {results_path}")

    df = _clean_results(results_path)
    plot_metric_table(df, out_dir)
    plot_progress(df, out_dir)
    plot_vram_runtime(df, out_dir)

    latest = _latest_successful_run(df, Path(args.runs_dir))
    if latest is not None:
        plot_history(latest, out_dir)
        plot_predictions(latest, out_dir)
        plot_router(latest, out_dir)

    print(f"Wrote plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
