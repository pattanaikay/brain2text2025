# plot_style.py  ──────────────────────────────────────────────────────────
# Drop-in style module. Import at the top of every notebook:
#   import plot_style
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 1. Clean base ────────────────────────────────────────────────────────────────
sns.set_style("ticks")            # removes gray background
sns.set_context("paper", font_scale=1.35)

plt.rcParams.update({
    # Typography
    "font.family":            "sans-serif",
    "font.sans-serif":        ["IBM Plex Sans", "Helvetica Neue", "Arial"],

    # Figure / output
    "figure.facecolor":       "white",
    "figure.dpi":             150,
    "savefig.dpi":            200,
    "savefig.bbox":           "tight",
    "savefig.facecolor":      "white",

    # Axes
    "axes.facecolor":         "white",
    "axes.edgecolor":         "#2D2D2D",
    "axes.linewidth":         0.8,
    "axes.spines.top":        False,    # <-- removes box globally
    "axes.spines.right":      False,
    "axes.titlesize":         14,
    "axes.titleweight":       "semibold",
    "axes.titlepad":          10,
    "axes.labelsize":         11,
    "axes.labelcolor":        "#333333",

    # Grid — whisper quiet
    "axes.grid":              True,
    "axes.grid.axis":         "x",      # "x" for hbar; "y" for vbar
    "grid.color":             "#EBEBEB",
    "grid.linewidth":         0.4,

    # Ticks
    "xtick.major.size":       4,
    "xtick.major.width":      0.8,
    "xtick.labelsize":        10,
    "ytick.labelsize":        10.5,
    "xtick.color":            "#555555",
    "ytick.color":            "#555555",

    # Legend
    "legend.frameon":         False,
    "legend.fontsize":        9.5,
    "legend.title_fontsize":  10,
    "legend.handlelength":    1.2,
    "legend.borderpad":       0,
    "legend.labelspacing":    0.4,
})

# 2. Colour palette ────────────────────────────────────────────────────────────
TRACK_COLORS = {
    "Encoder":    "#2563EB",   # blue
    "Loss":       "#D97706",   # amber
    "Projector":  "#059669",   # emerald
    "JEPA":       "#C2410C",   # brick red
    "Memory":     "#7C3AED",   # violet
    "Baseline":   "#94A3B8",   # slate
}

# 3. Per-figure helpers ────────────────────────────────────────────────────────
def clean_ax(ax, hbar=False):
    """Post-plot cleanup — call once after all ax.plot/barh/etc calls."""
    sns.despine(ax=ax)
    ax.set_axisbelow(True)          # grid renders behind bars/points
    if hbar:
        ax.tick_params(axis="y", length=0)  # no tick marks on category axis

def label_hbars(ax, fmt="{:.4f}", fontsize=9):
    """White value labels just inside horizontal bar ends."""
    for bar in ax.patches:
        w = bar.get_width()
        ax.text(
            w - abs(w) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(w),
            va="center", ha="right",
            color="white", fontsize=fontsize, fontweight="600",
        )

# 4. Example — Top-10 learning-slope chart ────────────────────────────────────
if __name__ == "__main__":
    names  = [d["name"]  for d in your_data]
    slopes = [d["slope"] for d in your_data]
    tracks = [d["track"] for d in your_data]

    colors = [TRACK_COLORS[t] for t in tracks]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(names, slopes, color=colors, height=0.65, zorder=2)

    # B0 reference line
    ax.axvline(x=-0.0062, color="#EF4444", lw=0.9,
               ls="--", alpha=0.6, zorder=1, label="B0 baseline")

    label_hbars(ax)
    clean_ax(ax, hbar=True)

    ax.set_title("Top experiments by learning slope", loc="left")
    ax.set_xlabel("Learning slope  (more negative = faster improvement)", labelpad=8)
    ax.legend(loc="lower right")

    plt.tight_layout(pad=1.5)
    plt.savefig("top10_clean.png")