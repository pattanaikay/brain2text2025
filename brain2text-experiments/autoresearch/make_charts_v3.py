"""
autoresearch/make_charts_v3.py
Publication-quality seaborn + plotly charts for all sweep tracks.

Usage:
    py -3 autoresearch/make_charts_v3.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY = True
except ImportError:
    PLOTLY = False
    print("[charts] plotly not installed — skipping HTML charts. Run: pip install plotly")

REPO = Path(__file__).resolve().parent.parent
DB   = REPO / "results" / "leaderboard.sqlite"
OUT  = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Design system ──────────────────────────────────────────────────────────────
# Track palette — Okabe-Ito colorblind-friendly
TRACK_HEX = {
    "B": "#0077BB",   # blue     — encoder
    "C": "#AA3377",   # magenta  — decoder
    "D": "#EE7733",   # orange   — loss
    "E": "#009988",   # teal     — projector
    "F": "#CC3311",   # red      — JEPA
    "A": "#BBBBBB",
    "H": "#BBBBBB",
}

LABEL_HEX = {
    "STRONG":    "#1565C0",
    "PROMISING": "#42A5F5",
    "WEAK":      "#FFA726",
    "INERT":     "#BDBDBD",
    "FAIL":      "#EF5350",
}

B0_SLOPE = -0.00617

# ── Global seaborn theme ───────────────────────────────────────────────────────
sns.set_theme(
    style="whitegrid",
    rc={
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelcolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#DDDDDD",
        "axes.linewidth": 0.8,
        "grid.color": "#EEEEEE",
        "grid.linewidth": 0.7,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    },
)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all() -> dict[str, dict]:
    """Return best (lowest slope) toy run per experiment id."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs "
        "WHERE profile='toy' ORDER BY expt_id, slope ASC"
    ).fetchall()
    conn.close()
    best: dict[str, dict] = {}
    for expt_id, w10, sl, bw in rows:
        if expt_id not in best or (
            sl is not None
            and (best[expt_id]["slope"] is None or sl < best[expt_id]["slope"])
        ):
            best[expt_id] = {"wer10": w10, "slope": sl, "best_wer": bw, "track": expt_id[0]}
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


# ── Shared helpers ─────────────────────────────────────────────────────────────
def _polish(ax: plt.Axes) -> None:
    """Remove top/right spines and soften the remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#DDDDDD")
    ax.tick_params(length=0)


def _ref_vline(ax: plt.Axes, x: float, label: str, color: str = "#555555") -> None:
    ax.axvline(x, color=color, linewidth=1.3, linestyle="--", label=label, zorder=2)


def _ref_hline(ax: plt.Axes, y: float, label: str, color: str = "#555555") -> None:
    ax.axhline(y, color=color, linewidth=1.3, linestyle="--", label=label, zorder=2)


def _label_legend(ax_or_fig, labels_present: list[str], **legend_kwargs) -> None:
    patches = [
        mpatches.Patch(color=LABEL_HEX[l], label=l)
        for l in ("STRONG", "PROMISING", "WEAK", "INERT", "FAIL")
        if l in labels_present
    ]
    if patches:
        ax_or_fig.legend(handles=patches, **legend_kwargs)


def _text_on(hex_color: str) -> str:
    """Choose readable text for a filled bar color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "white" if luminance < 0.58 else "#333333"


def _label_barh_centers(ax: plt.Axes, bars, labels: list[str], colors: list[str],
                        min_width: float = 0.002) -> None:
    """Place labels inside horizontal bars, with a small outside fallback."""
    x0, x1 = ax.get_xlim()
    span = abs(x1 - x0) or 1.0
    pad = span * 0.012

    for bar, label, color in zip(bars, labels, colors):
        width = bar.get_width()
        y = bar.get_y() + bar.get_height() / 2
        if abs(width) >= min_width:
            ax.text(width / 2, y, label, va="center", ha="center",
                    fontsize=8, color=_text_on(color), fontweight="bold",
                    clip_on=False)
        else:
            offset = pad if width >= 0 else -pad
            ha = "left" if width >= 0 else "right"
            ax.text(width + offset, y, label, va="center", ha=ha,
                    fontsize=8, color="#555555", clip_on=False)


def _label_bar_centers(ax: plt.Axes, bars, labels: list[str], colors: list[str]) -> None:
    """Place labels inside vertical bars when possible."""
    y0, y1 = ax.get_ylim()
    span = abs(y1 - y0) or 1.0
    pad = span * 0.02

    for bar, label, color in zip(bars, labels, colors):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        if abs(height) > span * 0.08:
            ax.text(x, height / 2, label, va="center", ha="center",
                    fontsize=8, color=_text_on(color), fontweight="bold")
        else:
            va = "bottom" if height >= 0 else "top"
            ax.text(x, height + (pad if height >= 0 else -pad), label,
                    va=va, ha="center", fontsize=8, color="#555555")


def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [static]  {name}.svg / .png")


def save_plotly(pfig, name: str) -> None:
    if not PLOTLY:
        return
    pfig.write_html(str(OUT / f"{name}.html"))
    print(f"  [html]    {name}.html")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 00 — Master slope x WER scatter (all tracks)
# ══════════════════════════════════════════════════════════════════════════════

def chart_master(data: dict) -> None:
    valid = {k: v for k, v in data.items() if v["slope"] is not None and v["wer10"] is not None}
    if not valid:
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot per-track so legend entries are clean
    for track in sorted({v["track"] for v in valid.values()}):
        items = [(eid, d) for eid, d in valid.items() if d["track"] == track]
        xs = [d["slope"] for _, d in items]
        ys = [d["wer10"] for _, d in items]
        lbs = [perf_label(d["slope"]) for _, d in items]
        sz = [160 if l == "STRONG" else (100 if l == "PROMISING" else 60) for l in lbs]
        ax.scatter(xs, ys, s=sz, color=TRACK_HEX.get(track, "#888888"),
                   edgecolors="white", linewidths=0.9, zorder=3, alpha=0.9,
                   label=f"Track {track}")
        for (eid, _), xi, yi in zip(items, xs, ys):
            ax.annotate(eid, (xi, yi), fontsize=7.5, ha="left",
                        xytext=(5, 3), textcoords="offset points", color="#555555")

    ax.axvline(0, color="#CCCCCC", linewidth=0.9, linestyle="--", zorder=1)
    ax.axvline(B0_SLOPE, color="#E05C5C", linewidth=1.4, linestyle=":",
               zorder=2, label=f"B0 baseline  slope={B0_SLOPE:.4f}")

    # "Better than baseline" annotation
    ylims = ax.get_ylim()
    ax.text(B0_SLOPE - 0.002, ylims[1] * 0.97,
            "  ← better than baseline",
            ha="right", va="top", fontsize=8, color="#1565C0", style="italic")

    ax.invert_yaxis()
    ax.set_xlabel("Slope  (Δ WER/epoch — more negative = faster improvement)")
    ax.set_ylabel("WER @ epoch 10  (lower = better)")
    ax.set_title(
        "Autoresearch Sweep — All Experiments\nSlope × WER@10, coloured by track",
        pad=14,
    )
    ax.legend(loc="upper right")
    _polish(ax)
    fig.tight_layout()
    save_fig(fig, "00_MASTER_scatter")

    if PLOTLY:
        xs_all = [d["slope"] for d in valid.values()]
        ys_all = [d["wer10"] for d in valid.values()]
        ids_all = list(valid.keys())
        tracks_all = [d["track"] for d in valid.values()]
        pfig = px.scatter(
            x=xs_all, y=ys_all, text=ids_all, color=tracks_all,
            color_discrete_map=TRACK_HEX,
            labels={"x": "Slope", "y": "WER@10", "color": "Track"},
            title="Brain2Text Autoresearch — All Experiments",
            template="plotly_white",
            height=580,
        )
        pfig.update_traces(
            textposition="top center",
            marker=dict(size=10, line=dict(color="white", width=1)),
        )
        pfig.update_yaxes(autorange="reversed")
        pfig.add_vline(x=0, line_dash="dash", line_color="#CCCCCC")
        pfig.add_vline(x=B0_SLOPE, line_dash="dot", line_color="#E05C5C",
                       annotation_text="B0 baseline", annotation_position="top right")
        pfig.update_layout(font_family="Arial", title_font_size=16)
        save_plotly(pfig, "00_MASTER_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 01 — Track B encoder sweep
# ══════════════════════════════════════════════════════════════════════════════

def chart_B(data: dict) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("B") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows:
        return

    ids     = sorted(rows.keys())
    wer10   = [rows[i]["wer10"] for i in ids]
    slopes  = [rows[i]["slope"] for i in ids]
    ref     = b0.get("slope", B0_SLOPE)
    labels_ = [perf_label(rows[i]["slope"], ref) for i in ids]
    colors  = [LABEL_HEX[l] for l in labels_]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Track B — Encoder Architecture\nvs B0_baseline (from-scratch BIT encoder)",
        fontsize=12, fontweight="bold",
    )

    # Left: WER@10
    ax = axes[0]
    bars = ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.6)
    if b0.get("wer10"):
        _ref_vline(ax, b0["wer10"], f"B0 = {b0['wer10']:.3f}")
    max_wer = max(wer10 + ([b0["wer10"]] if b0.get("wer10") else []))
    ax.set_xlim(max_wer * 1.10, 0)
    _label_barh_centers(
        ax, bars, [f"{val:.3f}  [{lbl}]" for val, lbl in zip(wer10, labels_)], colors,
        min_width=0.04,
    )
    ax.set_xlabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER @ Epoch 10")
    ax.legend()
    _polish(ax)

    # Right: slope
    ax = axes[1]
    bars = ax.barh(ids, slopes, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8)
    if b0.get("slope") is not None:
        _ref_vline(ax, b0["slope"], f"B0 slope = {b0['slope']:.4f}")
    _label_barh_centers(ax, bars, [f"{sl:.4f}" for sl in slopes], colors)
    ax.set_xlabel("Slope  (more negative = faster WER drop)")
    ax.set_title("Learning Slope (epochs 2→20)")
    ax.legend()
    _polish(ax)

    _label_legend(fig, labels_,
                  loc="lower center", ncol=len(set(labels_)),
                  bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    save_fig(fig, "01_B_encoder_sweep")

    if PLOTLY:
        pfig = make_subplots(rows=1, cols=2, subplot_titles=["WER@10", "Slope"])
        pfig.add_trace(
            go.Bar(y=ids, x=wer10, orientation="h", marker_color=colors,
                   text=[f"{w:.3f}" for w in wer10], textposition="outside", name="WER@10"),
            row=1, col=1,
        )
        pfig.add_trace(
            go.Bar(y=ids, x=slopes, orientation="h", marker_color=colors,
                   text=[f"{s:.4f}" for s in slopes], textposition="outside", name="Slope"),
            row=1, col=2,
        )
        pfig.update_layout(title="Track B — Encoder Architecture",
                           showlegend=False, height=500, template="plotly_white")
        pfig.update_xaxes(autorange="reversed", row=1, col=1)
        save_plotly(pfig, "01_B_encoder_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 02 — Track D loss sweep
# ══════════════════════════════════════════════════════════════════════════════

def chart_D(data: dict) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("D") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows:
        return

    ids     = sorted(rows.keys())
    slopes  = [rows[i]["slope"] for i in ids]
    wer10   = [rows[i]["wer10"] for i in ids]
    ref     = b0.get("slope", B0_SLOPE)
    labels_ = [perf_label(rows[i]["slope"], ref) for i in ids]
    colors  = [LABEL_HEX[l] for l in labels_]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Track D — Loss Function Design\nsame BIT+MLP+Qwen arch; only loss varies",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    bars = ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.55)
    if b0.get("wer10"):
        _ref_vline(ax, b0["wer10"], f"B0 = {b0['wer10']:.3f}")
    max_wer = max(wer10 + ([b0["wer10"]] if b0.get("wer10") else []))
    ax.set_xlim(max_wer * 1.10, 0)
    _label_barh_centers(ax, bars, [f"{val:.3f}" for val in wer10], colors, min_width=0.04)
    ax.set_xlabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER @ Epoch 10")
    ax.legend()
    _polish(ax)

    ax = axes[1]
    bars = ax.barh(ids, slopes, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8)
    if ref is not None:
        _ref_vline(ax, ref, f"B0 slope = {ref:.4f}")
    _label_barh_centers(ax, bars, [f"{sl:.4f}" for sl in slopes], colors)
    ax.set_xlabel("Slope  (more negative = better)")
    ax.set_title("Learning Slope")
    ax.legend()
    _polish(ax)

    _label_legend(fig, labels_,
                  loc="lower center", ncol=len(set(labels_)),
                  bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    save_fig(fig, "02_D_loss_sweep")

    if PLOTLY:
        pfig = make_subplots(rows=1, cols=2, subplot_titles=["WER@10", "Slope"])
        pfig.add_trace(
            go.Bar(y=ids, x=wer10, orientation="h", marker_color=colors,
                   text=[f"{w:.3f}" for w in wer10], textposition="outside", name="WER@10"),
            row=1, col=1,
        )
        pfig.add_trace(
            go.Bar(y=ids, x=slopes, orientation="h", marker_color=colors,
                   text=[f"{s:.4f}" for s in slopes], textposition="outside", name="Slope"),
            row=1, col=2,
        )
        pfig.update_layout(title="Track D — Loss Function Design",
                           showlegend=False, height=500, template="plotly_white")
        pfig.update_xaxes(autorange="reversed", row=1, col=1)
        save_plotly(pfig, "02_D_loss_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 03 — Track E projector sweep
# ══════════════════════════════════════════════════════════════════════════════

def chart_E(data: dict) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("E") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows:
        return

    ids     = sorted(rows.keys())
    slopes  = [rows[i]["slope"] for i in ids]
    wer10   = [rows[i]["wer10"] for i in ids]
    ref     = b0.get("slope", B0_SLOPE)
    labels_ = [perf_label(rows[i]["slope"], ref) for i in ids]
    colors  = [LABEL_HEX[l] for l in labels_]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Track E — Projector Architecture\nBIT encoder + Qwen decoder fixed; only projector varies",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    bars = ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.55)
    if b0.get("wer10"):
        _ref_vline(ax, b0["wer10"], f"B0 = {b0['wer10']:.3f}")
    max_wer = max(wer10 + ([b0["wer10"]] if b0.get("wer10") else []))
    ax.set_xlim(max_wer * 1.10, 0)
    _label_barh_centers(ax, bars, [f"{val:.3f}" for val in wer10], colors, min_width=0.04)
    ax.set_xlabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER @ Epoch 10")
    ax.legend()
    _polish(ax)

    ax = axes[1]
    bars = ax.barh(ids, slopes, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="#CCCCCC", linewidth=0.8)
    if ref is not None:
        _ref_vline(ax, ref, f"B0 slope = {ref:.4f}")
    _label_barh_centers(ax, bars, [f"{sl:.4f}" for sl in slopes], colors)
    ax.set_xlabel("Slope  (more negative = better)")
    ax.set_title("Learning Slope")
    ax.legend()
    _polish(ax)

    _label_legend(fig, labels_,
                  loc="lower center", ncol=len(set(labels_)),
                  bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    save_fig(fig, "03_E_projector_sweep")

    if PLOTLY:
        pfig = make_subplots(rows=1, cols=2, subplot_titles=["WER@10", "Slope"])
        pfig.add_trace(
            go.Bar(y=ids, x=wer10, orientation="h", marker_color=colors,
                   text=[f"{w:.3f}" for w in wer10], textposition="outside", name="WER@10"),
            row=1, col=1,
        )
        pfig.add_trace(
            go.Bar(y=ids, x=slopes, orientation="h", marker_color=colors,
                   text=[f"{s:.4f}" for s in slopes], textposition="outside", name="Slope"),
            row=1, col=2,
        )
        pfig.update_layout(title="Track E — Projector Architecture",
                           showlegend=False, height=420, template="plotly_white")
        pfig.update_xaxes(autorange="reversed", row=1, col=1)
        save_plotly(pfig, "03_E_projector_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 04 — Track C decoder (mostly failed)
# ══════════════════════════════════════════════════════════════════════════════

def chart_C(data: dict) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("C") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(
        "Track C — LLM Decoder Variants\nTarget: audio/multimodal LLM > text-only Qwen2.5",
        fontsize=12, fontweight="bold",
    )

    if rows:
        ids    = sorted(rows.keys())
        wer10  = [rows[i]["wer10"] for i in ids]
        ref    = b0.get("slope", B0_SLOPE)
        colors = [LABEL_HEX[perf_label(rows[i]["slope"], ref)] for i in ids]
        bars   = ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.5)
        if b0.get("wer10"):
            _ref_vline(ax, b0["wer10"], f"B0 text-only = {b0['wer10']:.3f}")
        max_wer = max(wer10 + ([b0["wer10"]] if b0.get("wer10") else []))
        ax.set_xlim(max_wer * 1.10, 0)
        _label_barh_centers(ax, bars, [f"{val:.3f}" for val in wer10], colors, min_width=0.04)
        ax.set_xlabel("WER @ epoch 10  (lower = better)")
        ax.legend()
    else:
        ax.set_facecolor("#FFF8F8")
        ax.text(0.5, 0.64, "C1 / C2 / C3 — all failed in this sweep",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color=LABEL_HEX["FAIL"], fontweight="bold")
        notes = (
            "C1: Qwen2Audio rejected as AutoModelForCausalLM\n"
            "C2: Phi4MM Flash Attention 2 crash  →  patched: attn_implementation=sdpa\n"
            "C3: Whisper bridge stayed on CPU  →  patched: bridge.to(device)"
        )
        ax.text(0.5, 0.30, notes, ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#666666", linespacing=1.7,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#FFF0F0",
                          edgecolor="#FFCCCC", linewidth=0.8))
        if b0.get("wer10"):
            ax.axvline(b0["wer10"], color="#555555", linewidth=1.3, linestyle="--",
                       label=f"B0 text-only = {b0['wer10']:.3f}")
            ax.set_xlim(0, 1.5)
        ax.set_xlabel("WER @ epoch 10")
        ax.legend()

    _polish(ax)
    fig.tight_layout()
    save_fig(fig, "04_C_decoder_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 05 — Track F JEPA pretraining
# ══════════════════════════════════════════════════════════════════════════════

def chart_F(data: dict) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("F") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows:
        return

    modality = {
        "F1": "Audio-JEPA\n(wav2vec2-style 1D)",
        "F2": "Video-JEPA\n(DINOv2-style 2D)",
        "F3": "Neural-JEPA\n(native patch-embed)",
    }
    ids    = [k for k in ("F1", "F2", "F3") if k in rows]
    names  = [modality.get(i, i) for i in ids]
    wer10  = [rows[i]["wer10"] for i in ids]
    slopes = [rows[i]["slope"] for i in ids]
    ref    = b0.get("slope", B0_SLOPE)
    colors = [LABEL_HEX[perf_label(rows[i]["slope"], ref)] for i in ids]

    x = np.arange(len(names))
    w = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Track F — JEPA Self-Supervised Pretraining\n"
        "Controlled A/B/C: identical specs except modality of JEPA pretraining",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    bars = ax.bar(x, wer10, width=w, color=colors, edgecolor="white")
    if b0.get("wer10"):
        _ref_hline(ax, b0["wer10"], f"B0 no-pretrain = {b0['wer10']:.3f}")
    ax.set_ylim(0, max(wer10 + ([b0["wer10"]] if b0.get("wer10") else [])) * 1.14)
    _label_bar_centers(
        ax, bars, [f"{val:.3f}\nslope={sl:.4f}" for val, sl in zip(wer10, slopes)], colors,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("WER @ epoch 10  (lower = better)")
    ax.set_title("Downstream Decoding WER")
    ax.legend()
    _polish(ax)

    ax = axes[1]
    bars2 = ax.bar(x, slopes, width=w, color=colors, edgecolor="white")
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    if ref is not None:
        _ref_hline(ax, ref, f"B0 slope = {ref:.4f}")
    ax.margins(y=0.18)
    _label_bar_centers(ax, bars2, [f"{sl:.4f}" for sl in slopes], colors)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Slope  (more negative = better)")
    ax.set_title("Learning Slope of Downstream Fine-tune")
    ax.legend()
    _polish(ax)

    labels_present = [perf_label(rows[i]["slope"], ref) for i in ids]
    _label_legend(fig, labels_present,
                  loc="lower center", ncol=len(set(labels_present)),
                  bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    save_fig(fig, "05_F_jepa_downstream")

    if PLOTLY:
        clean_names = [n.replace("\n", " ") for n in names]
        pfig = make_subplots(rows=1, cols=2,
                             subplot_titles=["Downstream WER@10", "Downstream Slope"])
        pfig.add_trace(
            go.Bar(x=clean_names, y=wer10, marker_color=colors,
                   text=[f"{w:.3f}" for w in wer10], textposition="outside", name="WER@10"),
            row=1, col=1,
        )
        pfig.add_trace(
            go.Bar(x=clean_names, y=slopes, marker_color=colors,
                   text=[f"{s:.4f}" for s in slopes], textposition="outside", name="Slope"),
            row=1, col=2,
        )
        pfig.update_layout(title="Track F — JEPA Pretraining",
                           showlegend=False, height=450, template="plotly_white")
        save_plotly(pfig, "05_F_jepa_downstream")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 06 — Combination phase
# ══════════════════════════════════════════════════════════════════════════════

def chart_combo(data: dict) -> None:
    def best_in(prefix):
        cands = {
            k: v for k, v in data.items()
            if k.startswith(prefix) and v["slope"] is not None and v["wer10"] is not None
        }
        return min(cands.items(), key=lambda x: x[1]["slope"]) if cands else None

    b_best = best_in("B")
    d_best = best_in("D")
    e_best = best_in("E")
    if not (b_best and e_best):
        return

    entries = []
    if b_best:
        entries.append((f"{b_best[0]}\n(Best Encoder)",   b_best[1]["wer10"], b_best[1]["slope"], "Encoder"))
    if d_best:
        entries.append((f"{d_best[0]}\n(Best Loss)",      d_best[1]["wer10"], d_best[1]["slope"], "Loss"))
    if e_best:
        entries.append((f"{e_best[0]}\n(Best Projector)", e_best[1]["wer10"], e_best[1]["slope"], "Projector"))
    for k, v in {k: v for k, v in data.items() if "CMB" in k.upper() and v["wer10"] is not None}.items():
        entries.append((k + "\n(Combined)", v["wer10"], v["slope"], "Combo"))

    ids    = [e[0] for e in entries]
    wer10  = [e[1] for e in entries]
    slopes = [e[2] for e in entries]
    types  = [e[3] for e in entries]
    type_colors = {
        "Encoder":  TRACK_HEX["B"],
        "Loss":     TRACK_HEX["D"],
        "Projector": TRACK_HEX["E"],
        "Combo":    "#FFB300",
    }
    colors = [type_colors.get(t, "#888888") for t in types]

    x = np.arange(len(ids))
    w = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Combination Phase — Do the Best Levers Stack?",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(x, wer10, width=w, color=colors, edgecolor="white")
    ax.set_ylim(0, max(wer10) * 1.12)
    _label_bar_centers(ax, bars, [f"{val:.3f}" for val in wer10], colors)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_ylabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER @ Epoch 10")
    _polish(ax)

    ax = axes[1]
    bars2 = ax.bar(x, slopes, width=w, color=colors, edgecolor="white")
    ax.axhline(0, color="#CCCCCC", linewidth=0.8)
    ax.margins(y=0.16)
    _label_bar_centers(ax, bars2, [f"{sl:.4f}" for sl in slopes], colors)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=8)
    ax.set_ylabel("Slope  (more negative = better)")
    ax.set_title("Learning Slope")
    _polish(ax)

    patches = [mpatches.Patch(color=v, label=k) for k, v in type_colors.items()
               if k in set(types)]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.88])
    save_fig(fig, "06_COMBO_phase")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 07 — Top-10 podium
# ══════════════════════════════════════════════════════════════════════════════

def chart_podium(data: dict) -> None:
    valid = [
        (k, v) for k, v in data.items()
        if v["slope"] is not None and v["wer10"] is not None
    ]
    top = sorted(valid, key=lambda x: x[1]["slope"])[:10]

    ids    = [t[0] for t in top]
    slopes = [t[1]["slope"] for t in top]
    wer10  = [t[1]["wer10"] for t in top]
    tracks = [t[1]["track"] for t in top]
    colors = [TRACK_HEX.get(tr, "#888888") for tr in tracks]

    # Rank labels: text medals for top 3
    rank_labels = {0: "1st", 1: "2nd", 2: "3rd"}
    y_labels = [f"{rank_labels.get(i, f'#{i+1}')}  {eid}" for i, eid in enumerate(ids)]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(ids))
    bars = ax.barh(y, slopes, color=colors, edgecolor="white", height=0.68)

    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=9.5)

    ax.axvline(0, color="#CCCCCC", linewidth=0.8)
    ax.axvline(B0_SLOPE, color="#E05C5C", linewidth=1.4, linestyle=":",
               label=f"B0 baseline ({B0_SLOPE:.4f})")
    left = min(slopes) * 1.05 if min(slopes) < 0 else min(slopes) * 0.95
    right = max(slopes) * 1.05 if max(slopes) > 0 else abs(min(slopes)) * 0.04
    ax.set_xlim(left, max(right, 0.001))
    _label_barh_centers(
        ax, bars, [f"{sl:.4f} | WER {wer:.3f}" for wer, sl in zip(wer10, slopes)], colors,
        min_width=0.003,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Slope  (more negative = faster WER drop = better)")
    ax.set_title("Top 10 Experiments by Slope — Cross-Track Podium",
                 fontsize=14, fontweight="bold")

    track_patches = [
        mpatches.Patch(color=TRACK_HEX.get(t, "#888888"), label=f"Track {t}")
        for t in sorted({v["track"] for _, v in top})
    ]
    fig.legend(
        handles=track_patches + [mpatches.Patch(color="#E05C5C", label="B0 baseline")],
        loc="lower center", ncol=len(track_patches) + 1,
        bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9,
    )
    _polish(ax)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig, "07_TOP10_podium")

    if PLOTLY:
        pfig = px.bar(
            x=slopes,
            y=[f"#{i+1} {eid}" for i, eid in enumerate(ids)],
            orientation="h",
            color=tracks,
            color_discrete_map=TRACK_HEX,
            labels={"x": "Slope", "y": "Experiment", "color": "Track"},
            title="Top 10 Experiments — Cross-Track Podium",
            template="plotly_white",
            height=480,
        )
        pfig.update_yaxes(autorange="reversed")
        pfig.add_vline(x=B0_SLOPE, line_dash="dot", line_color="#E05C5C",
                       annotation_text="B0 baseline")
        pfig.update_layout(font_family="Arial", title_font_size=16)
        save_plotly(pfig, "07_TOP10_podium")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Generating charts  ->  {OUT}/")
    print(f"Plotly available: {PLOTLY}\n")
    data = load_all()
    print(f"Loaded {len(data)} experiments from leaderboard.\n")

    chart_master(data)
    chart_B(data)
    chart_D(data)
    chart_E(data)
    chart_C(data)
    chart_F(data)
    chart_combo(data)
    chart_podium(data)

    print(f"\nAll charts saved to {OUT}/")
    if PLOTLY:
        print("Interactive HTML files also generated.")
