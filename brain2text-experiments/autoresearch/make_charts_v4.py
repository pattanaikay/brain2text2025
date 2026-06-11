"""
autoresearch/make_charts_v4.py
Redesigned sweep charts: ranked, fully-labeled, diverging slope bars.

Why this design (fixes for v2/v3):
  - On toy runs WER@10 saturates near 1.0 -> it has almost no signal. Slope is
    the one discriminating metric, so every chart is a RANKED slope chart and
    WER@10 is demoted to a compact right-hand column.
  - Every bar is labeled "ID · human name" (earlier versions left bars
    anonymous or relied on a colour legend alone).
  - Bars diverge around 0; verdict (STRONG/PROMISING/WEAK/INERT) is the colour;
    the B0 control slope is a single dashed reference line.
  - Explicit title layout so titles/subtitles/legends can never overlap data.

Usage:
    py -3 autoresearch/make_charts_v4.py
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DB   = REPO / "results" / "leaderboard.sqlite"
OUT  = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

B0_ID = "B0_baseline"

NAME = {
    "B0_baseline": "BIT from scratch (control)",
    "B1": "ConformerXL + jitter prenet",
    "B2": "HRM dual-timescale DEQ",
    "B3": "MambaPOSSM (GRU proxy)",
    "B4": "MoE encoder (6+2 experts)",
    "B5": "ZenBrain memory encoder",
    "C1": "Qwen2-Audio-7B decoder",
    "C2": "Phi-4-Multimodal decoder",
    "C3": "Whisper-Qwen split stack",
    "D1b": "CTC anneal 0.3→0",
    "D1d": "CTC removed",
    "D2a": "Contrastive removed",
    "D2d": "Contrastive ×2",
    "D3b": "TopoLoss λ=0.001",
    "D3c": "TopoLoss λ=0.01",
    "D4":  "Label smoothing ε=0.1",
    "E1a": "Deep MLP (5-layer)",
    "E1b": "Gated MLP",
    "E2b": "Q-Former (32 queries)",
    "E3":  "Patch×query grid best",
    "F1":  "Audio-JEPA backbone",
    "F2":  "Video-JEPA backbone",
    "F3":  "Neural-JEPA backbone",
    "H2":  "ZenBrain episodic E2E",
}

TRACK_NAME  = {"B": "Encoders", "C": "Decoders", "D": "Losses",
               "E": "Projectors", "F": "JEPA", "H": "Memory"}
TRACK_COLOR = {"B": "#3b6fb6", "C": "#2e9c6b", "D": "#e08214",
               "E": "#8062a8", "F": "#c5443c", "H": "#8c8c8c"}
VERDICT_COLOR = {"STRONG": "#1a5fa8", "PROMISING": "#6baed6",
                 "WEAK": "#fdbe6e", "INERT": "#c4c4c4"}

TEXT, MUTED, FAINT = "#222222", "#777777", "#bbbbbb"
REF_RED = "#c5443c"


def load() -> dict[str, dict]:
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT expt_id, wer_at_ep10, slope FROM runs "
        "WHERE profile='toy' AND slope IS NOT NULL AND wer_at_ep10 IS NOT NULL"
    ).fetchall()
    conn.close()
    best: dict[str, dict] = {}
    for eid, w10, sl in rows:
        if eid not in best or sl < best[eid]["slope"]:
            best[eid] = {"wer10": w10, "slope": sl, "track": eid[0]}
    return best


def verdict(slope: float, ref: float) -> str:
    if slope < ref - 0.004: return "STRONG"
    if slope < ref:          return "PROMISING"
    if slope < ref + 0.005:  return "WEAK"
    return "INERT"


def ranked_chart(items, ref, title, subtitle, fname,
                 note=None, color_by_track=False):
    """One ranked diverging-bar chart. items = [(expt_id, row)] best->worst."""
    n = len(items)
    fig_h = 2.1 + 0.52 * n + (0.35 if note else 0)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))

    ids    = [eid for eid, _ in items]
    slopes = [r["slope"] for _, r in items]
    wers   = [r["wer10"] for _, r in items]
    verds  = [verdict(s, ref) for s in slopes]
    if color_by_track:
        colors = [TRACK_COLOR.get(r["track"], "#888") for _, r in items]
    else:
        # B0 is the control, not a competitor — give it a neutral colour
        colors = ["#5a5a5a" if eid == B0_ID else VERDICT_COLOR[v]
                  for eid, v in zip(ids, verds)]

    ypos = list(range(n))[::-1]          # rank 1 at the top
    ax.barh(ypos, slopes, height=0.62, color=colors,
            edgecolor="white", linewidth=0.5, zorder=3)

    ax.axvline(0, color=FAINT, lw=1.0, zorder=2)
    ax.axvline(ref, color=REF_RED, lw=1.3, ls=(0, (4, 3)), zorder=2)

    xmin_d = min(min(slopes), ref)
    xmax_d = max(max(slopes), 0.0)
    any_neg = any(s < 0 for s in slopes)
    any_pos = any(s >= 0 for s in slopes)

    # Iteratively solve axis limits so that name labels (≈30 % of axis width,
    # drawn on the empty side of each bar), value labels and the WER column
    # always have room — regardless of whether bars go left, right, or both.
    L, R = xmin_d, xmax_d
    for _ in range(6):
        span = R - L
        L_new = xmin_d - 0.07 * span
        R_new = xmax_d + 0.15 * span          # value label + WER col past bars
        if any_neg:                            # their labels live right of 0
            R_new = max(R_new, 0.0 + 0.44 * span)
        if any_pos:                            # their labels live left of 0
            L_new = min(L_new, 0.0 - 0.34 * span)
        L, R = L_new, R_new
    span = R - L
    pad  = span * 0.012
    right_edge = R
    ax.set_xlim(L, R)

    for y, eid, sl, w10 in zip(ypos, ids, slopes, wers):
        lbl = f"{eid}  ·  {NAME.get(eid, eid)}"
        if sl < 0:   # bar grows left -> label to the right of the axis origin
            ax.text(pad, y, lbl, va="center", ha="left",
                    fontsize=9.5, color=TEXT, zorder=4)
            ax.text(sl - pad, y, f"{sl:+.4f}", va="center", ha="right",
                    fontsize=8.5, color=MUTED, zorder=4)
        else:        # bar grows right -> label to the left
            ax.text(-pad, y, lbl, va="center", ha="right",
                    fontsize=9.5, color=TEXT, zorder=4)
            ax.text(sl + pad, y, f"{sl:+.4f}", va="center", ha="left",
                    fontsize=8.5, color=MUTED, zorder=4)
        ax.text(right_edge - pad, y, f"{w10:.3f}", va="center", ha="right",
                fontsize=8.5, color=MUTED, zorder=4, family="monospace")

    ax.text(right_edge - pad, n - 0.25, "WER@10", va="bottom", ha="right",
            fontsize=8.5, color=MUTED, fontweight="bold")

    ax.set_yticks([])
    ax.set_ylim(-0.7, n - 0.3 + 0.55)
    ax.set_xlabel("Learning slope  (Δ WER per epoch — negative = improving)",
                  fontsize=10, color=TEXT)
    ax.tick_params(axis="x", labelsize=9, colors=MUTED)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(FAINT)
    ax.grid(axis="x", color="#eeeeee", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    # Explicit title block (cannot overlap)
    fig.text(0.012, 1 - 0.10 / fig_h, title, fontsize=15, fontweight="bold",
             color=TEXT, va="top", ha="left")
    fig.text(0.012, 1 - 0.44 / fig_h, subtitle, fontsize=9.5,
             color=MUTED, va="top", ha="left")

    if color_by_track:
        seen = []
        for _, r in items:
            if r["track"] not in seen:
                seen.append(r["track"])
        handles = [Patch(fc=TRACK_COLOR.get(t, "#888"),
                         label=f"{t} · {TRACK_NAME.get(t, t)}") for t in seen]
    else:
        present = []
        for eid, v in zip(ids, verds):
            if eid != B0_ID and v not in present:
                present.append(v)
        handles = [Patch(fc=VERDICT_COLOR[v], label=v.capitalize())
                   for v in present]
        if B0_ID in ids:
            handles.append(Patch(fc="#5a5a5a", label="B0 control"))
    handles.append(Line2D([], [], color=REF_RED, lw=1.3, ls=(0, (4, 3)),
                          label=f"B0 control slope {ref:+.4f}"))
    if color_by_track:
        # many rows -> no safe corner inside; put the legend below the axes
        fig.legend(handles=handles, loc="lower center", fontsize=8.5,
                   frameon=False, ncol=3,
                   bbox_to_anchor=(0.5, (0.34 if note else 0.06) / fig_h))
        extra_bottom = 1.30 if note else 1.05
    else:
        # bottom-left inside is empty: top rows bar left, bottom rows bar right
        ax.legend(handles=handles, loc="lower left", fontsize=8.5,
                  frameon=False, borderaxespad=0.4)
        extra_bottom = 0.92 if note else 0.62

    if note:
        fig.text(0.012, 0.08 / fig_h, note, fontsize=8.5, color=MUTED,
                 va="bottom", ha="left", style="italic")

    fig.subplots_adjust(left=0.02, right=0.985,
                        top=1 - 0.92 / fig_h,
                        bottom=extra_bottom / fig_h)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{fname}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  {fname}.png / .svg")


def main() -> None:
    data = load()
    ref  = data.get(B0_ID, {}).get("slope", -0.00617)
    print(f"Loaded {len(data)} experiments; B0 reference slope = {ref:+.5f}")
    print(f"Output -> {OUT}\\n")

    by_slope = sorted(data.items(), key=lambda kv: kv[1]["slope"])

    ranked_chart(
        by_slope, ref,
        "Autoresearch sweep — every experiment, ranked",
        "Toy profile (20 epochs × 200 batches, A100). Slope is the ranking metric "
        "— WER@10 saturates near 1.0 at this scale. Red dashes mark the B0 control.",
        "v4_00_overview_ranking",
        note="Not shown (failed/deferred): B3_mamba (CUDA build) · C1 Qwen2-Audio (loader) "
             "· C2 Phi-4-MM (attn impl) · A1–A3 (data/tooling).",
        color_by_track=True,
    )

    track_meta = {
        "B": ("Track B — Encoder architectures",
              "Projector + decoder fixed (MLP + Qwen2.5-1.5B). All encoders trained "
              "from scratch and judged against B0, the same-budget BIT control.",
              "B2 ran after the bf16 dtype fix but stayed INERT; B3_mamba skipped "
              "(CUDA 12.6 toolkit vs PyTorch cu130)."),
        "D": ("Track D — Loss recipes",
              "Identical BIT+MLP+Qwen stack; only the loss composition varies. "
              "CE always on; CTC / contrastive / TopoLoss / smoothing vary.",
              "D3b & D3c first crashed (TopoLoss blur kernel left on CPU); both "
              "re-run after the device fix — D3c became the sweep's best loss."),
        "E": ("Track E — Projector architectures",
              "Encoder (BIT) and decoder (Qwen2.5-1.5B) fixed; only the "
              "neural→text projector varies.", None),
        "F": ("Track F — JEPA pretrained backbones (downstream fine-tune)",
              "Controlled A/B/C — specs byte-identical except modality. Bars show "
              "downstream fine-tune of each pretrained backbone vs the no-pretraining "
              "B0 control.", None),
    }
    for i, (tr, (title, sub, note)) in enumerate(track_meta.items(), start=1):
        items = [(k, v) for k, v in by_slope if k.startswith(tr)]
        if not items:
            continue
        if tr != "B" and B0_ID in data:
            items = sorted(items + [(B0_ID, data[B0_ID])],
                           key=lambda kv: kv[1]["slope"])
        ranked_chart(items, ref, title, sub, f"v4_0{i}_track_{tr}", note=note)

    print("\nDone.")


if __name__ == "__main__":
    main()
