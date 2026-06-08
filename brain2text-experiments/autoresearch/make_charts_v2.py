"""
autoresearch/make_charts_v2.py
Seaborn + Plotly charts for all sweep tracks.
Generates both static (SVG/PNG via seaborn) and interactive (HTML via plotly).

Usage:
    python autoresearch/make_charts_v2.py
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

# Optional: plotly for interactive charts
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

# ── Seaborn theme ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = sns.color_palette("muted")

TRACK_COLOR = {
    "B": PALETTE[0],   # blue
    "C": PALETTE[2],   # green
    "D": PALETTE[1],   # orange
    "E": PALETTE[4],   # purple
    "F": PALETTE[3],   # red
    "A": PALETTE[5],   # brown
    "combo": PALETTE[7] if len(PALETTE) > 7 else "#888",
}

LABEL_COLOR = {"STRONG": "#2166ac", "PROMISING": "#74add1",
               "WEAK": "#fdae61",   "INERT": "#aaaaaa", "FAIL": "#d73027"}

B0_SLOPE = -0.00617  # B0_baseline reference

# ── Data loading ──────────────────────────────────────────────────────────────

def load_all() -> dict[str, dict]:
    """Return {expt_id: {wer_at_ep10, slope, best_wer, track}} for best toy row per expt."""
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs "
        "WHERE profile='toy' ORDER BY expt_id, slope ASC NULLS LAST"
    ).fetchall()
    conn.close()
    best: dict[str, dict] = {}
    for expt_id, w10, sl, bw in rows:
        # keep the row with most-negative slope (best) per experiment
        if expt_id not in best or (sl is not None and (best[expt_id]["slope"] is None or sl < best[expt_id]["slope"])):
            best[expt_id] = {"wer10": w10, "slope": sl, "best_wer": bw, "track": expt_id[0]}
    return best

def label(slope: float | None, ref: float = B0_SLOPE) -> str:
    if slope is None: return "FAIL"
    delta = slope - ref
    if slope < ref - 0.004: return "STRONG"
    if slope < ref:          return "PROMISING"
    if slope < ref + 0.005:  return "WEAK"
    return "INERT"


# ── Save helper ───────────────────────────────────────────────────────────────

def save_fig(fig: plt.Figure, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  [static]  {name}.svg / .png")

def save_plotly(fig, name: str) -> None:
    if not PLOTLY: return
    fig.write_html(str(OUT / f"{name}.html"))
    print(f"  [interactive] {name}.html")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Master slope×WER scatter (all tracks)
# ══════════════════════════════════════════════════════════════════════════════

def chart_master(data: dict[str, dict]) -> None:
    valid = {k: v for k, v in data.items() if v["slope"] is not None and v["wer10"] is not None}
    if not valid: return

    df_x = [v["slope"] for v in valid.values()]
    df_y = [v["wer10"]  for v in valid.values()]
    ids  = list(valid.keys())
    tracks = [v["track"] for v in valid.values()]
    labels = [label(v["slope"]) for v in valid.values()]

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (eid, xi, yi, tr) in enumerate(zip(ids, df_x, df_y, tracks)):
        c = TRACK_COLOR.get(tr, "#888")
        lbl = label(xi)
        ms  = 120 if lbl == "STRONG" else (80 if lbl == "PROMISING" else 55)
        ax.scatter(xi, yi, color=c, s=ms, edgecolors="white", linewidths=0.6,
                   zorder=3, alpha=0.9)
        ax.annotate(eid, (xi, yi), fontsize=7.5, ha="left",
                    xytext=(5, 2), textcoords="offset points", color="#444")

    ax.axvline(0, color="#cccccc", linewidth=0.8, linestyle="--")
    ax.axvline(B0_SLOPE, color="#e05c5c", linewidth=1.1, linestyle=":",
               label=f"B0 baseline slope ({B0_SLOPE:.4f})")
    ax.invert_yaxis()
    ax.set_xlabel("Slope (ΔWer/epoch — more negative = faster improvement)", fontsize=10)
    ax.set_ylabel("WER @ epoch 10 (lower = better)", fontsize=10)
    ax.set_title("Autoresearch Sweep — All Experiments\nSlope × WER@10, coloured by track",
                 fontsize=12, pad=12)

    patches = [mpatches.Patch(color=TRACK_COLOR.get(t, "#888"), label=f"Track {t}")
               for t in sorted(set(tracks))]
    ax.legend(handles=patches, fontsize=9, frameon=False, loc="upper right")
    sns.despine(ax=ax, left=False, bottom=False, top=True, right=True)
    fig.tight_layout()
    save_fig(fig, "00_MASTER_scatter")

    # Plotly version
    if PLOTLY:
        pfig = px.scatter(
            x=df_x, y=df_y, text=ids,
            color=tracks,
            color_discrete_map={t: f"rgb{tuple(int(c*255) for c in TRACK_COLOR.get(t, (0.5,0.5,0.5)))}"
                                 for t in set(tracks)},
            labels={"x": "Slope", "y": "WER@10", "color": "Track"},
            title="Brain2Text Autoresearch — All Experiments",
            height=600,
        )
        pfig.update_traces(textposition="top center", marker=dict(size=10))
        pfig.update_yaxes(autorange="reversed")
        pfig.add_vline(x=0, line_dash="dash", line_color="#ccc")
        pfig.add_vline(x=B0_SLOPE, line_dash="dot", line_color="#e05c5c",
                       annotation_text="B0 baseline")
        save_plotly(pfig, "00_MASTER_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Track B encoder sweep
# ══════════════════════════════════════════════════════════════════════════════

def chart_B(data: dict[str, dict]) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("B") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows: return

    ids    = sorted(rows.keys())
    wer10  = [rows[i]["wer10"]  for i in ids]
    slopes = [rows[i]["slope"]  for i in ids]
    labels_= [label(rows[i]["slope"], b0.get("slope", B0_SLOPE)) for i in ids]
    colors = [LABEL_COLOR[l] for l in labels_]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Track B — Encoder Architecture\n(vs B0_baseline from-scratch BIT)",
                 fontsize=12, fontweight="bold", y=1.01)

    # Left: WER@10 horizontal bar
    ax = axes[0]
    bars = ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.6)
    if b0.get("wer10"):
        ax.axvline(b0["wer10"], color="#555", linewidth=1.2, linestyle="--",
                   label=f"B0 WER@10 = {b0['wer10']:.3f}")
    for bar, lbl in zip(bars, labels_):
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height()/2,
                lbl, va="center", fontsize=7.5,
                color=LABEL_COLOR.get(lbl, "#888"))
    ax.invert_xaxis()
    ax.set_xlabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER @ epoch 10")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    # Right: slope bar, reference line
    ax = axes[1]
    bar_colors = [LABEL_COLOR[l] for l in labels_]
    ax.barh(ids, slopes, color=bar_colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    if b0.get("slope") is not None:
        ax.axvline(b0["slope"], color="#555", linewidth=1.2, linestyle="--",
                   label=f"B0 slope = {b0['slope']:.4f}")
    ax.set_xlabel("Slope  (more negative = faster WER drop)")
    ax.set_title("Learning slope (epoch 2→20)")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    save_fig(fig, "01_B_encoder_sweep")

    # Plotly
    if PLOTLY:
        pfig = make_subplots(rows=1, cols=2, subplot_titles=["WER@10", "Slope"])
        pfig.add_trace(go.Bar(y=ids, x=wer10, orientation="h", marker_color=colors,
                              name="WER@10"), row=1, col=1)
        pfig.add_trace(go.Bar(y=ids, x=slopes, orientation="h", marker_color=colors,
                              name="Slope"), row=1, col=2)
        pfig.update_layout(title="Track B — Encoder Architecture", showlegend=False, height=500)
        pfig.update_xaxes(autorange="reversed", row=1, col=1)
        save_plotly(pfig, "01_B_encoder_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Track D loss sweep
# ══════════════════════════════════════════════════════════════════════════════

def chart_D(data: dict[str, dict]) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("D") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows: return

    ids    = sorted(rows.keys())
    slopes = [rows[i]["slope"]  for i in ids]
    wer10  = [rows[i]["wer10"]  for i in ids]
    ref    = b0.get("slope", B0_SLOPE)
    labels_= [label(rows[i]["slope"], ref) for i in ids]
    colors = [LABEL_COLOR[l] for l in labels_]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Track D — Loss Function Design\n(same BIT+MLP+Qwen arch; only loss varies)",
                 fontsize=12, fontweight="bold")

    y = np.arange(len(ids))
    bars = ax.barh(y, slopes, color=colors, edgecolor="white", height=0.55)
    ax.set_yticks(y); ax.set_yticklabels(ids)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.axvline(ref, color="#555", linewidth=1.2, linestyle="--",
               label=f"B0 baseline slope ({ref:.4f})")

    for bar, lbl, wer in zip(bars, labels_, wer10):
        x_pos = bar.get_width()
        ax.text(x_pos + (0.001 if x_pos >= 0 else -0.001),
                bar.get_y() + bar.get_height()/2,
                f"{lbl}  WER@10={wer:.3f}",
                va="center", ha="left" if x_pos >= 0 else "right",
                fontsize=7.5, color=LABEL_COLOR.get(lbl, "#888"))

    ax.set_xlabel("Slope  (more negative = better)")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig(fig, "02_D_loss_sweep")

    if PLOTLY:
        pfig = px.bar(x=slopes, y=ids, orientation="h", color=labels_,
                      color_discrete_map=LABEL_COLOR,
                      labels={"x": "Slope", "y": "Experiment", "color": "Label"},
                      title="Track D — Loss Function Design", height=420)
        pfig.add_vline(x=ref, line_dash="dash", line_color="#555",
                       annotation_text=f"B0 ref ({ref:.4f})")
        save_plotly(pfig, "02_D_loss_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Track E projector sweep
# ══════════════════════════════════════════════════════════════════════════════

def chart_E(data: dict[str, dict]) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("E") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows: return

    ids    = sorted(rows.keys())
    slopes = [rows[i]["slope"]  for i in ids]
    wer10  = [rows[i]["wer10"]  for i in ids]
    ref    = b0.get("slope", B0_SLOPE)
    labels_= [label(rows[i]["slope"], ref) for i in ids]
    colors = [LABEL_COLOR[l] for l in labels_]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Track E — Projector Architecture\n(BIT encoder + Qwen decoder fixed)",
                 fontsize=12, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.55)
    if b0.get("wer10"):
        ax.axvline(b0["wer10"], color="#555", linewidth=1.2, linestyle="--",
                   label=f"B0 WER@10 = {b0['wer10']:.3f}")
    ax.invert_xaxis()
    ax.set_xlabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER @ epoch 10")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    ax = axes[1]
    ax.barh(ids, slopes, color=colors, edgecolor="white", height=0.55)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.axvline(ref, color="#555", linewidth=1.2, linestyle="--",
               label=f"B0 slope ({ref:.4f})")
    for i, (sl, lbl) in enumerate(zip(slopes, labels_)):
        ax.text(sl + 0.001, i, lbl, va="center", fontsize=7.5,
                color=LABEL_COLOR.get(lbl, "#888"))
    ax.set_xlabel("Slope  (more negative = better)")
    ax.set_title("Learning slope")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    save_fig(fig, "03_E_projector_sweep")

    if PLOTLY:
        pfig = px.bar(x=slopes, y=ids, orientation="h", color=labels_,
                      color_discrete_map=LABEL_COLOR, text=wer10,
                      labels={"x": "Slope", "y": "Projector", "color": "Label"},
                      title="Track E — Projector Architecture", height=380)
        pfig.update_traces(texttemplate="WER@10=%{text:.3f}", textposition="outside")
        save_plotly(pfig, "03_E_projector_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — Track C decoder (mostly failed, show gap + baseline)
# ══════════════════════════════════════════════════════════════════════════════

def chart_C(data: dict[str, dict]) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("C") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Track C — LLM Decoder Variants\n(target: audio/multimodal LLM > text-only Qwen)",
                 fontsize=12, fontweight="bold")

    if rows:
        ids   = sorted(rows.keys())
        wer10 = [rows[i]["wer10"] for i in ids]
        ref   = b0.get("slope", B0_SLOPE)
        colors = [LABEL_COLOR[label(rows[i]["slope"], ref)] for i in ids]
        ax.barh(ids, wer10, color=colors, edgecolor="white", height=0.5)
        if b0.get("wer10"):
            ax.axvline(b0["wer10"], color="#555", linewidth=1.2, linestyle="--",
                       label=f"B0 text-only baseline = {b0['wer10']:.3f}")
        ax.invert_xaxis()
        ax.set_xlabel("WER @ epoch 10  (lower = better)")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.55, "C1 / C2 / C3 — all failed in this sweep",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="#d73027", fontweight="bold")
        ax.text(0.5, 0.38,
                "C1: Qwen2Audio can't load as AutoModelForCausalLM\n"
                "C2: Phi4MM rejected Flash Attention 2 (fix: attn_implementation=sdpa — patched)\n"
                "C3: Whisper bridge linear on CPU (fix: bridge.to(device) — patched)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="#666", style="italic")
        if b0.get("wer10"):
            ax.axvline(b0["wer10"], color="#555", linewidth=1.2, linestyle="--",
                       label=f"B0 text-only baseline = {b0['wer10']:.3f}")
            ax.set_xlim(0, 1.5)
        ax.set_xlabel("WER @ epoch 10")

    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig(fig, "04_C_decoder_sweep")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Track F JEPA (downstream comparison)
# ══════════════════════════════════════════════════════════════════════════════

def chart_F(data: dict[str, dict]) -> None:
    rows = {k: v for k, v in data.items() if k.startswith("F") and v["wer10"] is not None}
    b0   = data.get("B0_baseline", {})
    if not rows: return

    order = {"F1": "Audio\n(wav2vec2-style 1D)", "F2": "Video\n(DINOv2-style 2D)",
             "F3": "Neural\n(native patch-embed)"}
    ids   = [k for k in ("F1", "F2", "F3") if k in rows]
    names = [order.get(i, i) for i in ids]
    wer10 = [rows[i]["wer10"]  for i in ids]
    slopes= [rows[i]["slope"]  for i in ids]
    ref   = b0.get("slope", B0_SLOPE)
    colors = [LABEL_COLOR[label(rows[i]["slope"], ref)] for i in ids]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle("Track F — JEPA Self-Supervised Pretraining\n"
                 "Controlled A/B/C: specs identical except modality",
                 fontsize=12, fontweight="bold", y=1.02)

    ax = axes[0]
    bars = ax.bar(names, wer10, color=colors, edgecolor="white", width=0.45)
    for bar, sl in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"slope\n{sl:.4f}", ha="center", fontsize=8, color="#555")
    if b0.get("wer10"):
        ax.axhline(b0["wer10"], color="#555", linewidth=1.2, linestyle="--",
                   label=f"B0 no-pretraining = {b0['wer10']:.3f}")
    ax.set_ylabel("WER @ epoch 10")
    ax.set_title("Downstream decoding WER")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    ax = axes[1]
    ax.bar(names, slopes, color=colors, edgecolor="white", width=0.45)
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.axhline(ref, color="#555", linewidth=1.2, linestyle="--",
               label=f"B0 slope ({ref:.4f})")
    ax.set_ylabel("Slope  (more negative = better)")
    ax.set_title("Learning slope of downstream fine-tune")
    ax.legend(fontsize=8, frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()
    save_fig(fig, "05_F_jepa_downstream")

    if PLOTLY:
        pfig = make_subplots(rows=1, cols=2, subplot_titles=["Downstream WER@10", "Downstream Slope"])
        pfig.add_trace(go.Bar(x=names, y=wer10, marker_color=colors, name="WER@10"), row=1, col=1)
        pfig.add_trace(go.Bar(x=names, y=slopes, marker_color=colors, name="Slope"), row=1, col=2)
        pfig.update_layout(title="Track F — JEPA Pretraining (audio vs video vs neural)",
                           showlegend=False, height=420)
        save_plotly(pfig, "05_F_jepa_downstream")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — Combination phase
# ══════════════════════════════════════════════════════════════════════════════

def chart_combo(data: dict[str, dict]) -> None:
    # Compare solo winners vs combination
    # Pull best per track
    def best_in(prefix):
        cands = {k: v for k, v in data.items()
                 if k.startswith(prefix) and v["slope"] is not None and v["wer10"] is not None}
        return min(cands.items(), key=lambda x: x[1]["slope"]) if cands else None

    b_best = best_in("B")
    d_best = best_in("D")
    e_best = best_in("E")

    # Find combo row — any B1 run that has override info (slope > usual)
    combo_rows = {k: v for k, v in data.items() if k == "B1" and v["wer10"] is not None}

    if not (b_best and e_best): return

    entries = []
    if b_best:  entries.append((f"{b_best[0]}\n(Best Encoder)",   b_best[1]["wer10"],  b_best[1]["slope"],  "Encoder"))
    if d_best:  entries.append((f"{d_best[0]}\n(Best Loss)",      d_best[1]["wer10"],  d_best[1]["slope"],  "Loss"))
    if e_best:  entries.append((f"{e_best[0]}\n(Best Projector)", e_best[1]["wer10"],  e_best[1]["slope"],  "Projector"))

    # Add combo if it exists separately
    cmb = {k: v for k, v in data.items() if "CMB" in k.upper() and v["wer10"] is not None}
    for k, v in cmb.items():
        entries.append((k + "\n(Combined)", v["wer10"], v["slope"], "Combo"))

    ids    = [e[0] for e in entries]
    wer10  = [e[1] for e in entries]
    slopes = [e[2] for e in entries]
    types  = [e[3] for e in entries]
    colors = [LABEL_COLOR[label(s)] for s in slopes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Combination Phase — Do the Best Levers Stack?",
                 fontsize=12, fontweight="bold", y=1.02)

    ax = axes[0]
    ax.bar(ids, wer10, color=colors, edgecolor="white", width=0.45)
    ax.set_ylabel("WER @ epoch 10  (lower = better)")
    ax.set_title("WER@10 comparison")
    ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax)

    ax = axes[1]
    ax.bar(ids, slopes, color=colors, edgecolor="white", width=0.45)
    ax.axhline(0, color="#aaa", linewidth=0.8)
    ax.set_ylabel("Slope  (more negative = better)")
    ax.set_title("Slope comparison")
    ax.tick_params(axis="x", labelsize=8)
    sns.despine(ax=ax)

    fig.tight_layout()
    save_fig(fig, "06_COMBO_phase")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Summary podium (top 6 across all tracks)
# ══════════════════════════════════════════════════════════════════════════════

def chart_podium(data: dict[str, dict]) -> None:
    valid = [(k, v) for k, v in data.items() if v["slope"] is not None and v["wer10"] is not None]
    top = sorted(valid, key=lambda x: x[1]["slope"])[:10]

    ids    = [t[0] for t in top]
    slopes = [t[1]["slope"]  for t in top]
    wer10  = [t[1]["wer10"]  for t in top]
    tracks = [t[1]["track"]  for t in top]
    colors = [TRACK_COLOR.get(tr, "#888") for tr in tracks]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(range(len(ids)), slopes, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"#{i+1}  {eid}" for i, eid in enumerate(ids)], fontsize=9)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.axvline(B0_SLOPE, color="#e05c5c", linewidth=1.1, linestyle=":",
               label=f"B0 baseline ({B0_SLOPE:.4f})")

    for bar, wer in zip(bars, wer10):
        ax.text(bar.get_width() - 0.001, bar.get_y() + bar.get_height()/2,
                f"WER@10={wer:.3f}", va="center", ha="right",
                fontsize=7.5, color="white", fontweight="bold")

    ax.invert_yaxis()
    ax.set_xlabel("Slope  (more negative = better)")
    ax.set_title("Top 10 Experiments by Slope — Cross-Track Podium", fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=TRACK_COLOR.get(t, "#888"), label=f"Track {t}")
               for t in sorted({v["track"] for _, v in top})]
    ax.legend(handles=patches + [mpatches.Patch(color="#e05c5c", label="B0 baseline")],
              fontsize=8, frameon=False, loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    save_fig(fig, "07_TOP10_podium")

    if PLOTLY:
        pfig = px.bar(
            x=slopes, y=[f"#{i+1} {eid}" for i, eid in enumerate(ids)],
            orientation="h",
            color=tracks,
            hover_data={"WER@10": wer10},
            labels={"x": "Slope", "y": "Experiment", "color": "Track"},
            title="Top 10 Experiments — Cross-Track Podium",
            height=450,
        )
        pfig.update_yaxes(autorange="reversed")
        pfig.add_vline(x=B0_SLOPE, line_dash="dot", line_color="#e05c5c",
                       annotation_text="B0 baseline")
        save_plotly(pfig, "07_TOP10_podium")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Generating charts → {OUT}/")
    print(f"Plotly available: {PLOTLY}")
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
