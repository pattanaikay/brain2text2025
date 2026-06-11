"""
autoresearch/make_charts_v3.py

Polished static PNG charts for the Brain2Text autoresearch sweep.

Usage:
    py -3 autoresearch/make_charts_v3.py
"""
from __future__ import annotations

import re
import sqlite3
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import plot_style                    # sets rcParams globally
from plot_style import TRACK_COLORS, clean_ax, label_hbars  # pull in the colour dict + helpers
DB = REPO / "results" / "leaderboard.sqlite"
REGISTRY = REPO / "registry.yaml"
OUT = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

B0_SLOPE = -0.00617

TRACK_NAMES = {
    "A": "Analysis",
    "B": "Encoder",
    "C": "Decoder",
    "D": "Loss",
    "E": "Projector",
    "F": "JEPA",
    "G": "Adaptation",
    "H": "Memory",
}

TRACK_HEX = {
    "B": "#0072B2",  # Okabe-Ito Blue
    "C": "#CC79A7",  # Okabe-Ito Reddish Purple
    "D": "#E69F00",  # Okabe-Ito Orange
    "E": "#009E73",  # Okabe-Ito Bluish Green (Teal)
    "F": "#D55E00",  # Okabe-Ito Vermilion (Red)
    "G": "#56B4E9",  # Okabe-Ito Sky Blue
    "H": "#9D755D",  # Brown
    "A": "#000000",  # Black
}

LABEL_HEX = {
    "STRONG": "#2F6DB3",
    "PROMISING": "#65A9D9",
    "WEAK": "#E9A33A",
    "INERT": "#B8BBC1",
    "FAIL": "#D95F5F",
}

DISPLAY_ALIASES = {
    "B0_baseline": "BIT baseline",
    "B1": "ConformerXL",
    "B2": "HRM dual-timescale",
    "B3": "MambaPOSSM (GRU)",
    "B4": "MoE (6 experts)",
    "B5": "ZenBrain memory",
    "D1b": "CTC anneal",
    "D1d": "No CTC",
    "D2a": "No contrastive",
    "D2d": "Contrastive x2",
    "D3b": "TopoLoss 0.001",
    "D3c": "TopoLoss 0.01",
    "D4": "Label smoothing",
    "E1a": "Deep MLP",
    "E1b": "Gated MLP",
    "E2b": "Q-Former 32",
    "E3": "Patch x Q-Former grid",
    "F1": "Audio-JEPA",
    "F2": "Video-JEPA",
    "F3": "Neural-JEPA",
    "H2": "ZenBrain E2E",
}

sns.set_theme(
    context="paper",
    style="white",
    palette="colorblind",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D9DDE3",
        "axes.labelcolor": "#30343B",
        "axes.labelsize": 10,
        "axes.titlecolor": "#242830",
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#EBEBEB",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.color": "#555B64",
        "xtick.labelsize": 9,
        "ytick.color": "#555B64",
        "ytick.labelsize": 9,
    },
)


def load_registry_names() -> dict[str, str]:
    """Return experiment names from registry.yaml without requiring PyYAML."""
    if not REGISTRY.exists():
        return {}

    try:
        import yaml  # type: ignore

        with REGISTRY.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return {
            expt_id: meta.get("name", expt_id)
            for expt_id, meta in (raw.get("experiments") or {}).items()
            if isinstance(meta, dict)
        }
    except Exception:
        names: dict[str, str] = {}
        current_id: str | None = None
        in_experiments = False
        with REGISTRY.open("r", encoding="utf-8") as f:
            for line in f:
                if re.match(r"^experiments:\s*$", line):
                    in_experiments = True
                    continue
                if not in_experiments:
                    continue
                if re.match(r"^\S", line):
                    break
                m_id = re.match(r"^  ([A-Za-z0-9_]+):\s*$", line)
                if m_id:
                    current_id = m_id.group(1)
                    continue
                m_name = re.match(r'^\s{4}name:\s*["\']?(.*?)["\']?\s*$', line)
                if current_id and m_name:
                    names[current_id] = m_name.group(1)
        return names


EXPERIMENT_NAMES = load_registry_names()


def sort_key(expt_id: str) -> tuple:
    parts = re.split(r"(\d+)", expt_id)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def display_name(expt_id: str, width: int = 30) -> str:
    name = DISPLAY_ALIASES.get(expt_id, EXPERIMENT_NAMES.get(expt_id, expt_id))
    text = f"{name} ({expt_id})"
    return textwrap.fill(text, width=width, break_long_words=False)


def compact_name(expt_id: str, width: int = 22) -> str:
    name = DISPLAY_ALIASES.get(expt_id, EXPERIMENT_NAMES.get(expt_id, expt_id))
    return textwrap.fill(f"{name}\n({expt_id})", width=width, break_long_words=False)


def load_all() -> dict[str, dict]:
    """Return best toy row per experiment id, ranked by lowest slope."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs "
        "WHERE profile='toy' ORDER BY expt_id, slope ASC"
    ).fetchall()
    conn.close()

    best: dict[str, dict] = {}
    for expt_id, w10, slope, best_wer in rows:
        if expt_id not in best or (
            slope is not None
            and (best[expt_id]["slope"] is None or slope < best[expt_id]["slope"])
        ):
            best[expt_id] = {
                "id": expt_id,
                "name": EXPERIMENT_NAMES.get(expt_id, expt_id),
                "display": display_name(expt_id),
                "compact": compact_name(expt_id),
                "wer10": w10,
                "slope": slope,
                "best_wer": best_wer,
                "track": expt_id[0],
                "track_name": TRACK_NAMES.get(expt_id[0], f"Track {expt_id[0]}"),
            }
    return best


def perf_label(slope: float | None, ref: float = B0_SLOPE) -> str:
    if slope is None:
        return "FAIL"
    if slope < ref - 0.004:
        return "STRONG"
    if slope < ref:
        return "PROMISING"
    if slope < ref + 0.005:
        return "WEAK"
    return "INERT"


def _text_on(hex_color: str) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.58 else "#25303B"


def save_fig(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [png] {name}.png")


def _finish(ax: plt.Axes) -> None:
    ax.tick_params(length=0)
    sns.despine(ax=ax, left=False, bottom=False)


def _label_perf_legend(fig_or_ax, labels_present: list[str], **kwargs) -> None:
    handles = [
        mpatches.Patch(color=LABEL_HEX[label], label=label.title())
        for label in ("STRONG", "PROMISING", "WEAK", "INERT", "FAIL")
        if label in labels_present
    ]
    if handles:
        fig_or_ax.legend(handles=handles, **kwargs)


def _ref_vline(ax: plt.Axes, x: float, label: str, color: str = "#6B7178") -> None:
    ax.axvline(x, color=color, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(
        x,
        0.98,
        label,
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8,
        color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5),
    )


def _ref_hline(ax: plt.Axes, y: float, label: str, color: str = "#6B7178") -> None:
    ax.axhline(y, color=color, linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)


def _rows(data: dict, prefix: str) -> list[dict]:
    return [
        data[k]
        for k in sorted(data, key=sort_key)
        if k.startswith(prefix) and data[k]["wer10"] is not None
    ]


def _sweep_frame(data: dict, prefix: str) -> list[dict]:
    b0 = data.get("B0_baseline", {})
    ref = b0.get("slope", B0_SLOPE)
    rows = _rows(data, prefix)
    for row in rows:
        row["label"] = perf_label(row["slope"], ref)
        row["color"] = LABEL_HEX[row["label"]]
    return rows


def _bar_labels(ax: plt.Axes, values: list[float], colors: list[str], fmt: str) -> None:
    x0, x1 = ax.get_xlim()
    is_inverted = x0 > x1
    span = abs(x1 - x0) or 1.0
    min_inside_width = span * 0.055
    pad = span * 0.012

    for patch, value, color in zip(ax.patches, values, colors):
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        if abs(width) < min_inside_width:
            ax.text(
                width + pad,
                y,
                fmt.format(value),
                ha="right" if is_inverted else "left",
                va="center",
                fontsize=8,
                color=color,
                fontweight="medium",
            )
            continue

        ax.text(
            width / 2,
            y,
            fmt.format(value),
            ha="center",
            va="center",
            fontsize=8,
            color=_text_on(color),
            fontweight="medium",
        )


def _colored_barplot(ax: plt.Axes, values: list[float], y_labels: list[str], colors: list[str]) -> None:
    """Draw ordered horizontal bars and color them explicitly."""
    sns.barplot(x=values, y=y_labels, color="#C7CBD1", ax=ax, orient="h")
    for patch, color in zip(ax.patches, colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.9)


def _hide_repeated_y_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)


def _metric_sweep_chart(
    data: dict,
    prefix: str,
    title: str,
    note: str,
    output: str,
    height: float = 5.0,
) -> None:
    rows = _sweep_frame(data, prefix)
    b0 = data.get("B0_baseline", {})
    if not rows:
        return

    y_labels = [r["display"] for r in rows]
    wer10 = [r["wer10"] for r in rows]
    slopes = [r["slope"] for r in rows]
    colors = [r["color"] for r in rows]
    labels = [r["label"] for r in rows]
    ref = b0.get("slope", B0_SLOPE)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, height),
        gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.08},
        sharey=True,
    )
    fig.suptitle(title, fontsize=15, fontweight="normal", x=0.02, y=1.04, ha="left")
    fig.text(0.02, 0.98, note, fontsize=9, color="#666E78", ha="left")

    ax = axes[0]
    _colored_barplot(ax, wer10, y_labels, colors)
    clean_ax(ax, hbar=True)
    max_wer = max(wer10 + ([b0["wer10"]] if b0.get("wer10") is not None else []))
    ax.set_xlim(max_wer * 1.08, 0)
    if b0.get("wer10") is not None:
        _ref_vline(ax, b0["wer10"], f"B0 {b0['wer10']:.3f}")
    _bar_labels(ax, wer10, colors, "{:.3f}")
    ax.set_title("WER at epoch 10", loc="left", pad=8)
    ax.set_xlabel("WER, lower is better")
    ax.set_ylabel("")
    _finish(ax)

    ax = axes[1]
    _colored_barplot(ax, slopes, y_labels, colors)
    clean_ax(ax, hbar=True)
    ax.axvline(0, color="#CDD2D8", linewidth=0.9)
    x_min = min(slopes + ([ref] if ref is not None else []))
    x_max = max(slopes + [0])
    pad = max((x_max - x_min) * 0.10, 0.003)
    ax.set_xlim(x_min - pad, x_max + pad)
    if ref is not None:
        _ref_vline(ax, ref, f"B0 {ref:.4f}")
    _bar_labels(ax, slopes, colors, "{:.4f}")
    ax.set_title("Learning slope", loc="left", pad=8)
    ax.set_xlabel("Delta WER per epoch, more negative is better")
    _hide_repeated_y_axis(ax)
    _finish(ax)

    _label_perf_legend(
        fig,
        labels,
        loc="lower center",
        ncol=max(1, len(set(labels))),
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    save_fig(fig, output)


def chart_master(data: dict) -> None:
    rows = [v for v in data.values() if v["slope"] is not None and v["wer10"] is not None]
    if not rows:
        return

    xs = [r["slope"] for r in rows]
    ys = [r["wer10"] for r in rows]
    tracks = [r["track_name"] for r in rows]
    labels = [perf_label(r["slope"]) for r in rows]
    sizes = [120 if l == "STRONG" else 85 if l == "PROMISING" else 55 for l in labels]
    track_palette = {
        TRACK_NAMES.get(track, f"Track {track}"): TRACK_HEX.get(track, "#888888")
        for track in sorted({r["track"] for r in rows})
    }

    fig, ax = plt.subplots(figsize=(13.5, 7))
    sns.scatterplot(
        x=xs,
        y=ys,
        hue=tracks,
        size=sizes,
        sizes=(55, 140),
        palette=track_palette,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.92,
        legend=False,
        ax=ax,
    )
    clean_ax(ax, hbar=False)

    # Label only the best performers and the baseline to keep the scatter airy.
    top_ids = {r["id"] for r in sorted(rows, key=lambda r: r["slope"])[:8]}
    top_ids.add("B0_baseline")
    for row in rows:
        if row["id"] in top_ids:
            ax.annotate(
                row["id"],
                (row["slope"], row["wer10"]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color="#4B535D",
            )

    ax.axvline(0, color="#CDD2D8", linewidth=0.9, linestyle=(0, (4, 3)))
    ax.axvline(B0_SLOPE, color="#D76565", linewidth=1.2, linestyle=(0, (2, 2)))
    ax.invert_yaxis()
    ax.set_title("Autoresearch sweep landscape", loc="left", pad=10)
    ax.text(
        0,
        1.01,
        "Slope vs WER at epoch 10. Labeled points are the strongest runs plus B0.",
        transform=ax.transAxes,
        fontsize=9,
        color="#666E78",
        ha="left",
    )
    ax.set_xlabel("Learning slope, more negative is better")
    ax.set_ylabel("WER at epoch 10, lower is better")

    present_tracks = sorted({r["track"] for r in rows})
    handles = [
        mpatches.Patch(color=TRACK_HEX.get(t, "#888888"), label=TRACK_NAMES.get(t, f"Track {t}"))
        for t in present_tracks
    ]
    handles.append(mpatches.Patch(color="#D76565", label=f"B0 slope {B0_SLOPE:.4f}"))
    ax.legend(handles=handles, loc="upper right", title=None)
    _finish(ax)
    fig.tight_layout()
    save_fig(fig, "00_MASTER_scatter")


def chart_B(data: dict) -> None:
    _metric_sweep_chart(
        data,
        "B",
        "Encoder sweep",
        "Architecture variants compared against the BIT baseline.",
        "01_B_encoder_sweep",
        height=5.4,
    )


def chart_D(data: dict) -> None:
    _metric_sweep_chart(
        data,
        "D",
        "Loss sweep",
        "Same encoder/projector/decoder stack; only the loss recipe varies.",
        "02_D_loss_sweep",
        height=5.8,
    )


def chart_E(data: dict) -> None:
    _metric_sweep_chart(
        data,
        "E",
        "Projector sweep",
        "Deep MLP, gated MLP, and Q-Former designs for the neural-to-text bridge.",
        "03_E_projector_sweep",
        height=4.9,
    )


def chart_C(data: dict) -> None:
    rows = _sweep_frame(data, "C")
    b0 = data.get("B0_baseline", {})

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.set_title("Decoder sweep", loc="left", pad=10)
    ax.text(
        0,
        1.01,
        "Audio and multimodal decoders were attempted against the text-only baseline.",
        transform=ax.transAxes,
        fontsize=9,
        color="#666E78",
        ha="left",
    )

    if rows:
        y_labels = [r["display"] for r in rows]
        wer10 = [r["wer10"] for r in rows]
        labels = [r["label"] for r in rows]
        colors = [r["color"] for r in rows]
        _colored_barplot(ax, wer10, y_labels, colors)
        max_wer = max(wer10 + ([b0["wer10"]] if b0.get("wer10") is not None else []))
        ax.set_xlim(max_wer * 1.08, 0)
        if b0.get("wer10") is not None:
            _ref_vline(ax, b0["wer10"], f"B0 {b0['wer10']:.3f}")
        _bar_labels(ax, wer10, colors, "{:.3f}")
        ax.set_ylabel("")
        _label_perf_legend(
            fig,
            labels,
            loc="lower center",
            ncol=max(1, len(set(labels))),
            bbox_to_anchor=(0.5, -0.03),
        )
    else:
        ax.set_facecolor("#FFF9F9")
        ax.text(
            0.5,
            0.62,
            "Decoder variants did not produce valid toy rows",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color=LABEL_HEX["FAIL"],
        )
        notes = (
            "Qwen2-Audio: model class mismatch\n"
            "Phi-4-Multimodal: attention backend retry pending\n"
            "Whisper-Qwen: bridge/device retry pending"
        )
        ax.text(
            0.5,
            0.34,
            notes,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="#666E78",
            linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#FFF0F0", edgecolor="#F0C9C9"),
        )
        if b0.get("wer10") is not None:
            _ref_vline(ax, b0["wer10"], f"B0 {b0['wer10']:.3f}")
            ax.set_xlim(0, 1.5)

    ax.set_xlabel("WER at epoch 10")
    _finish(ax)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, "04_C_decoder_sweep")


def chart_F(data: dict) -> None:
    _metric_sweep_chart(
        data,
        "F",
        "JEPA pretraining sweep",
        "Downstream runs after audio, video, or native neural self-supervised pretraining.",
        "05_F_jepa_downstream",
        height=4.9,
    )


def chart_combo(data: dict) -> None:
    def best_in(prefix: str):
        candidates = [
            (k, v)
            for k, v in data.items()
            if k.startswith(prefix) and v["slope"] is not None and v["wer10"] is not None
        ]
        return min(candidates, key=lambda item: item[1]["slope"]) if candidates else None

    entries = []
    for label, prefix in (("Best encoder", "B"), ("Best loss", "D"), ("Best projector", "E")):
        best = best_in(prefix)
        if best:
            expt_id, row = best
            entries.append(
                {
                    "id": expt_id,
                    "name": f"{DISPLAY_ALIASES.get(expt_id, EXPERIMENT_NAMES.get(expt_id, expt_id))}\n({label})",
                    "wer10": row["wer10"],
                    "slope": row["slope"],
                    "kind": label,
                    "color": TRACK_HEX.get(prefix, "#888888"),
                }
            )

    for expt_id, row in data.items():
        if "CMB" in expt_id.upper() and row["wer10"] is not None:
            entries.append(
                {
                    "id": expt_id,
                    "name": f"{display_name(expt_id, 24)}\n(combined)",
                    "wer10": row["wer10"],
                    "slope": row["slope"],
                    "kind": "Combined",
                    "color": "#E6AB02",
                }
            )

    if not entries:
        return

    names = [e["name"] for e in entries]
    wer10 = [e["wer10"] for e in entries]
    slopes = [e["slope"] for e in entries]
    colors = [e["color"] for e in entries]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"wspace": 0.32})
    fig.suptitle("Best building blocks", fontsize=15, fontweight="normal", x=0.02, y=1.04, ha="left")
    fig.text(0.02, 0.98, "A quick check on whether the strongest individual levers stack.", fontsize=9, color="#666E78")

    ax = axes[0]
    sns.barplot(x=names, y=wer10, hue=names, palette=dict(zip(names, colors)), legend=False, ax=ax)
    clean_ax(ax, hbar=False)
    ax.set_ylim(0, max(wer10) * 1.12)
    for patch, value, color in zip(ax.patches, wer10, colors):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() / 2,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=_text_on(color),
            fontsize=8,
            fontweight="medium",
        )
    ax.set_title("WER at epoch 10", loc="left", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("WER, lower is better")
    _finish(ax)

    ax = axes[1]
    sns.barplot(x=names, y=slopes, hue=names, palette=dict(zip(names, colors)), legend=False, ax=ax)
    clean_ax(ax, hbar=False)
    ax.axhline(0, color="#CDD2D8", linewidth=0.9)
    ax.margins(y=0.18)
    for patch, value, color in zip(ax.patches, slopes, colors):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value / 2,
            f"{value:.4f}",
            ha="center",
            va="center",
            color=_text_on(color),
            fontsize=8,
            fontweight="medium",
        )
    ax.set_title("Learning slope", loc="left", pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Delta WER per epoch")
    _finish(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, "06_COMBO_phase")


def chart_podium(data: dict) -> None:
    valid = [v for v in data.values() if v["slope"] is not None and v["wer10"] is not None]
    top = sorted(valid, key=lambda row: row["slope"])[:10]
    if not top:
        return

    labels = [f"#{idx + 1}  {display_name(row['id'], 34)}" for idx, row in enumerate(top)]
    slopes = [row["slope"] for row in top]
    wer10 = [row["wer10"] for row in top]
    colors = [TRACK_HEX.get(row["track"], "#888888") for row in top]

    fig, ax = plt.subplots(figsize=(12, 6.4))
    sns.barplot(x=slopes, y=labels, hue=labels, palette=dict(zip(labels, colors)), legend=False, ax=ax)
    clean_ax(ax, hbar=True)
    ax.axvline(0, color="#CDD2D8", linewidth=0.9)
    ax.axvline(B0_SLOPE, color="#D76565", linewidth=1.2, linestyle=(0, (2, 2)))

    left = min(slopes) * 1.06 if min(slopes) < 0 else min(slopes) * 0.94
    right = max(slopes) * 1.06 if max(slopes) > 0 else abs(min(slopes)) * 0.04
    ax.set_xlim(left, max(right, 0.001))
    for patch, slope, wer, color in zip(ax.patches, slopes, wer10, colors):
        ax.text(
            slope / 2,
            patch.get_y() + patch.get_height() / 2,
            f"{slope:.4f} | WER {wer:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color=_text_on(color),
            fontweight="medium",
        )

    ax.set_title("Top experiments by learning slope", loc="left", pad=10)
    ax.text(
        0,
        1.01,
        "Ranked across all sweep tracks. More negative slope means faster improvement.",
        transform=ax.transAxes,
        fontsize=9,
        color="#666E78",
        ha="left",
    )
    ax.set_xlabel("Learning slope, more negative is better")
    ax.set_ylabel("")

    present_tracks = sorted({row["track"] for row in top})
    handles = [
        mpatches.Patch(color=TRACK_HEX.get(track, "#888888"), label=TRACK_NAMES.get(track, track))
        for track in present_tracks
    ]
    handles.append(mpatches.Patch(color="#D76565", label="B0 baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.01))
    _finish(ax)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_fig(fig, "07_TOP10_podium")


if __name__ == "__main__":
    print(f"Generating PNG charts -> {OUT}/")
    data = load_all()
    print(f"Loaded {len(data)} experiments from leaderboard.")
    print(f"Loaded {len(EXPERIMENT_NAMES)} experiment names from registry.\n")

    chart_master(data)
    chart_B(data)
    chart_D(data)
    chart_E(data)
    chart_C(data)
    chart_F(data)
    chart_combo(data)
    chart_podium(data)

    print(f"\nAll PNG charts saved to {OUT}/")
