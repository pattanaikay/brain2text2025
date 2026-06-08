"""
autoresearch/make_charts.py
Generate Tufte-style seaborn charts for all sweep tracks.
Reads results/leaderboard.sqlite; writes results/figures/*.{svg,png}.

Usage:
    python autoresearch/make_charts.py
"""
import os
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO  = Path(__file__).resolve().parent.parent
DB    = REPO / "results" / "leaderboard.sqlite"
FIGS  = REPO / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Tufte palette: muted, high-contrast ──────────────────────────────────────
CLR = {
    "blue":   "#2166ac",
    "red":    "#d6604d",
    "green":  "#4dac26",
    "orange": "#f4a442",
    "purple": "#762a83",
    "grey":   "#aaaaaa",
    "dark":   "#333333",
}

def tufte_ax(ax):
    """Strip chart junk: remove top/right spines, lighten grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#555555")

def load_rows(where=""):
    conn = sqlite3.connect(DB)
    q = f"""
        SELECT expt_id, profile, wer_at_ep10, slope, best_wer
        FROM runs WHERE profile='toy' {where}
        ORDER BY expt_id, rowid DESC
    """
    rows = conn.execute(q).fetchall()
    conn.close()
    # deduplicate: keep latest per expt_id
    seen, out = {}, []
    for r in rows:
        if r[0] not in seen:
            seen[r[0]] = r
            out.append(r)
    return out

def save(fig, name):
    for ext in ("svg", "png"):
        p = FIGS / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {FIGS}/{name}.svg/.png")

# ── TRACK B: Encoder sweep ───────────────────────────────────────────────────
def chart_B():
    rows = [r for r in load_rows() if r[0].startswith("B") and r[2] is not None]
    if not rows:
        print("Track B: no data"); return

    b0 = next((r for r in rows if r[0] == "B0_baseline"), None)
    b0_slope = b0[3] if b0 else 0.0

    ids     = [r[0] for r in rows]
    wer10   = [r[2] for r in rows]
    slopes  = [r[3] for r in rows]
    colors  = [CLR["blue"] if s < b0_slope else CLR["grey"] for s in slopes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Track B — Encoder Architecture", fontsize=13, color=CLR["dark"], weight="bold")

    # Left: WER@10 bar
    bars = ax1.barh(ids, wer10, color=colors, edgecolor="white", height=0.6)
    if b0:
        ax1.axvline(b0[2], color=CLR["red"], linewidth=1.2, linestyle="--", label=f"B0 baseline WER@10={b0[2]:.3f}")
        ax1.legend(fontsize=8, frameon=False)
    ax1.set_xlabel("WER @ epoch 10 (lower = better)", fontsize=9)
    ax1.invert_xaxis()
    tufte_ax(ax1)

    # Right: slope scatter (size ∝ 1/best_wer proxy)
    sc = ax2.scatter(slopes, wer10, c=[CLR["blue"] if s < b0_slope else CLR["grey"] for s in slopes],
                     s=80, edgecolors="white", linewidths=0.5, zorder=3)
    for i, r in enumerate(rows):
        ax2.annotate(r[0], (slopes[i], wer10[i]), fontsize=7, ha="left",
                     xytext=(4, 0), textcoords="offset points", color="#555555")
    if b0:
        ax2.axvline(b0_slope, color=CLR["red"], linewidth=1, linestyle="--")
    ax2.set_xlabel("Slope (Δ WER/epoch; more negative = faster improvement)", fontsize=9)
    ax2.set_ylabel("WER @ epoch 10", fontsize=9)
    ax2.invert_yaxis()
    tufte_ax(ax2)

    plt.tight_layout()
    save(fig, "B_encoder_sweep")

# ── TRACK D: Loss sweep ───────────────────────────────────────────────────────
def chart_D():
    rows = [r for r in load_rows() if r[0].startswith("D") and r[2] is not None]
    b0   = next((r for r in load_rows() if r[0] == "B0_baseline" and r[2] is not None), None)
    if not rows:
        print("Track D: no data"); return

    b0_slope = b0[3] if b0 else 0.0
    ids    = [r[0] for r in rows]
    wer10  = [r[2] for r in rows]
    slopes = [r[3] for r in rows]
    colors = [CLR["blue"] if s < b0_slope else CLR["grey"] for s in slopes]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("Track D — Loss Function Design\n(vs BIT+default-loss baseline)", fontsize=13, color=CLR["dark"], weight="bold")

    ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.6)
    if b0:
        ax.axvline(b0[2], color=CLR["red"], linewidth=1.2, linestyle="--",
                   label=f"B0 baseline WER@10={b0[2]:.3f}")
    # annotate INERT ties
    for i, (sl, w) in enumerate(zip(slopes, wer10)):
        lbl = "STRONG" if sl < b0_slope - 0.004 else ("INERT" if sl > 0.01 else "")
        if lbl:
            ax.text(w + 0.002, i, lbl, va="center", fontsize=7,
                    color=CLR["blue"] if lbl == "STRONG" else CLR["grey"])

    ax.invert_xaxis()
    ax.set_xlabel("WER @ epoch 10", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    tufte_ax(ax)
    plt.tight_layout()
    save(fig, "D_loss_sweep")

# ── TRACK E: Projector sweep ──────────────────────────────────────────────────
def chart_E():
    rows = [r for r in load_rows() if r[0].startswith("E") and r[0] != "E3" and r[2] is not None]
    e3_rows = [r for r in load_rows() if r[0].startswith("E3") and r[2] is not None]
    b0      = next((r for r in load_rows() if r[0] == "B0_baseline" and r[2] is not None), None)

    if not rows and not e3_rows:
        print("Track E: no data"); return

    b0_slope = b0[3] if b0 else 0.0
    all_rows = rows + e3_rows
    ids    = [r[0] for r in all_rows]
    wer10  = [r[2] for r in all_rows]
    slopes = [r[3] for r in all_rows]
    colors = [CLR["blue"] if s < b0_slope else CLR["grey"] for s in slopes]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Track E — Projector Architecture", fontsize=13, color=CLR["dark"], weight="bold")

    ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.6)
    if b0:
        ax.axvline(b0[2], color=CLR["red"], linewidth=1.2, linestyle="--",
                   label=f"B0 baseline WER@10={b0[2]:.3f}")
    ax.invert_xaxis()
    ax.set_xlabel("WER @ epoch 10", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    tufte_ax(ax)
    plt.tight_layout()
    save(fig, "E_projector_sweep")

# ── TRACK F: JEPA pretraining ─────────────────────────────────────────────────
def chart_F():
    rows = [r for r in load_rows() if r[0].startswith("F") and r[2] is not None]
    b0   = next((r for r in load_rows() if r[0] == "B0_baseline" and r[2] is not None), None)
    if not rows:
        print("Track F: no JEPA downstream data"); return

    labels = {"F1": "Audio-JEPA", "F2": "Video-JEPA", "F3": "Neural-JEPA"}
    ids    = [labels.get(r[0], r[0]) for r in rows]
    wer10  = [r[2] for r in rows]
    slopes = [r[3] for r in rows]
    b0_slope = b0[3] if b0 else 0.0
    colors = [CLR["green"] if s < b0_slope else CLR["grey"] for s in slopes]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.suptitle("Track F — JEPA Backbone (downstream WER)\nF1=audio F2=video F3=neural (controlled A/B/C)",
                 fontsize=11, color=CLR["dark"], weight="bold")

    bars = ax.bar(ids, wer10, color=colors, edgecolor="white", width=0.5)
    for bar, sl in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"slope={sl:.3f}", ha="center", fontsize=8, color="#555555")
    if b0:
        ax.axhline(b0[2], color=CLR["red"], linewidth=1.2, linestyle="--",
                   label=f"B0 baseline WER@10={b0[2]:.3f}")

    ax.set_ylabel("WER @ epoch 10", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    tufte_ax(ax)
    plt.tight_layout()
    save(fig, "F_jepa_downstream")

# ── MASTER: Slope scatter across all tracks ───────────────────────────────────
def chart_master():
    rows = [r for r in load_rows() if r[2] is not None and r[3] is not None]
    if not rows:
        print("Master: no data"); return

    track_color = {
        "B": CLR["blue"], "D": CLR["orange"], "E": CLR["green"],
        "C": CLR["purple"], "F": CLR["red"],  "A": CLR["grey"], "H": CLR["grey"],
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Full Sweep — Slope vs WER@10 (all tracks)", fontsize=13,
                 color=CLR["dark"], weight="bold")

    for r in rows:
        track = r[0][0]
        c = track_color.get(track, CLR["grey"])
        ax.scatter(r[3], r[2], color=c, s=70, edgecolors="white",
                   linewidths=0.5, zorder=3, alpha=0.85)
        ax.annotate(r[0], (r[3], r[2]), fontsize=7, ha="left",
                    xytext=(4, 0), textcoords="offset points", color="#555555")

    ax.axvline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Slope (more negative = faster WER improvement)", fontsize=10)
    ax.set_ylabel("WER @ epoch 10 (lower = better)", fontsize=10)
    ax.invert_yaxis()

    patches = [mpatches.Patch(color=v, label=f"Track {k}")
               for k, v in track_color.items() if k not in ("H",)]
    ax.legend(handles=patches, fontsize=8, frameon=False, loc="upper right")
    tufte_ax(ax)
    plt.tight_layout()
    save(fig, "MASTER_slope_scatter")

# ── TRACK C: Decoder sweep (even if all failed — show known gap) ──────────────
def chart_C():
    rows = [r for r in load_rows() if r[0].startswith("C") and r[2] is not None]
    b0   = next((r for r in load_rows() if r[0] == "B0_baseline" and r[2] is not None), None)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.suptitle("Track C — LLM Decoder Variants\n(target: beat text-only Qwen2.5-1.5B baseline)",
                 fontsize=11, color=CLR["dark"], weight="bold")

    if rows:
        ids    = [r[0] for r in rows]
        wer10  = [r[2] for r in rows]
        b0_slope = b0[3] if b0 else 0.0
        colors = [CLR["purple"] if r[3] < b0_slope else CLR["grey"] for r in rows]
        ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.5)
    else:
        ax.text(0.5, 0.5, "C1/C2/C3 all failed in this sweep\n(environment bugs — see fixes doc)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color=CLR["red"], style="italic")

    if b0:
        ax.axvline(b0[2], color=CLR["red"], linewidth=1.2, linestyle="--",
                   label=f"B0 text-only baseline WER@10={b0[2]:.3f}")
    ax.invert_xaxis()
    ax.set_xlabel("WER @ epoch 10", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    tufte_ax(ax)
    plt.tight_layout()
    save(fig, "C_decoder_sweep")

# ── COMBINATION chart ─────────────────────────────────────────────────────────
def chart_combo():
    combo_rows = [r for r in load_rows() if r[0].startswith("B1") and r[2] is not None]
    b1_base = next((r for r in combo_rows), None)
    if not b1_base:
        print("Combo: no B1 data"); return

    # Pull solo-winner results for comparison
    winners = {}
    for r in load_rows():
        if r[0] in ("B1", "D1b", "D3c", "E1b") and r[2] is not None:
            if r[0] not in winners or r[3] < winners[r[0]][3]:
                winners[r[0]] = r

    labels  = list(winners.keys()) + ["CMB-1 (B1+QFormer+CTC)"]
    values  = [winners[k][2] for k in winners.keys()]
    cmb_rows = [r for r in load_rows() if "CMB" in r[0].upper() and r[2] is not None]
    values += [r[2] for r in cmb_rows[:1]] if cmb_rows else [1.0]
    colors  = [CLR["blue"]] * len(winners) + [CLR["orange"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Combination Phase — Winners vs Composed Stack",
                 fontsize=12, color=CLR["dark"], weight="bold")
    ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylabel("WER @ epoch 10", fontsize=9)
    ax.tick_params(axis="x", rotation=20)
    tufte_ax(ax)
    plt.tight_layout()
    save(fig, "COMBO_combination_phase")

if __name__ == "__main__":
    print(f"Generating charts → {FIGS}/")
    chart_B()
    chart_C()
    chart_D()
    chart_E()
    chart_F()
    chart_master()
    chart_combo()
    print("Done.")
