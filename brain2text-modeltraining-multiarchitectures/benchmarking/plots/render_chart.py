"""Render the Tufte-style supertable comparing six encoder architectures."""
import json

data = [
    # name, label, params_M, latency_ms, flops_T800_G, pattern, note
    ("hrm",       "HRM",          2.92,  551.5, 0.59,  "sequential 2-clock",  "O(1) memory via DEQ 1-step grad"),
    ("bit",       "BIT (base)",  10.72,  112.2, 2.14,  "parallel attn",       "RoPE, 7 layers; the reference point"),
    ("zenbrain",  "ZenBrain",    14.86,  138.0, 3.04,  "parallel + buffer",   "Adds cross-attn over episodic buffer"),
    ("mamba",     "Mamba",       16.83,   30.2, 2.49,  "sequential scan",     "Linear in T; GRU fallback on CPU"),
    ("conformer", "Conformer",   42.29,  283.5, 7.48,  "parallel + conv",     "12 macaron blocks, depthwise conv"),
    ("moe",       "MoE",         49.34,  222.5, 5.44,  "parallel sparse",     "Top-2 of 6 experts + 2 shared"),
]
# Sort by params ascending so the eye reads small → large capacity
data.sort(key=lambda r: r[2])

# ---- Canvas ----
W, H = 920, 560
M_TOP = 100
ROW_H = 60
N = len(data)

# Column x-bounds
X_LABEL_R = 175            # arch label right-justified to this x
X_C1_L, X_C1_R = 195, 365  # params panel
X_C2_L, X_C2_R = 395, 565  # latency panel
X_C3_L, X_C3_R = 595, 765  # FLOPs panel
X_NOTE_L = 790             # note column starts here

# Scales (range-frame: max → maxv, min → 0)
MAX_P = 50.0   # M params
MAX_L = 580.0  # ms latency
MAX_F = 8.0    # G FLOPs

def bar(x0, x1, value, vmax):
    return x0 + (x1 - x0) * (value / vmax)

# ---- SVG ----
svg = []
svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'font-family="Georgia, \'ET Book\', serif" font-size="13" fill="#222">')
svg.append('<rect width="100%" height="100%" fill="#fffff8"/>')

# Title block
svg.append(f'<text x="40" y="38" font-size="22" font-weight="500">'
           f'Six neural encoders on a 4-second toy stream</text>')
svg.append(f'<text x="40" y="60" font-size="13" font-style="italic" fill="#555">'
           f'B=4, T=200 bins (≈4 s), 512 channels. Latency is numpy-simulated '
           f'forward pass (median of 5–20 runs).</text>')
svg.append(f'<text x="40" y="78" font-size="12" fill="#555">'
           f'Rows sorted by parameter count. FLOPs column extrapolates to '
           f'T=800 bins (16 s) — the long end of real BCI streams.</text>')

# Column headers
svg.append(f'<text x="{X_LABEL_R}" y="{M_TOP - 12}" text-anchor="end" '
           f'font-size="12" fill="#444">architecture</text>')
svg.append(f'<text x="{X_C1_L}" y="{M_TOP - 12}" font-size="12" fill="#444">'
           f'parameters (M)</text>')
svg.append(f'<text x="{X_C2_L}" y="{M_TOP - 12}" font-size="12" fill="#444">'
           f'sim latency (ms)</text>')
svg.append(f'<text x="{X_C3_L}" y="{M_TOP - 12}" font-size="12" fill="#444">'
           f'FLOPs at T=800 (G)</text>')
svg.append(f'<text x="{X_NOTE_L}" y="{M_TOP - 12}" font-size="12" fill="#444">'
           f'compute pattern</text>')

# Rows
for i, (_name, label, p, lat, fl, pattern, note) in enumerate(data):
    y_mid = M_TOP + ROW_H * i + ROW_H / 2
    y_bar_top = y_mid - 9
    y_bar_h = 18

    # Architecture label, right-justified
    svg.append(f'<text x="{X_LABEL_R}" y="{y_mid + 4}" text-anchor="end" '
               f'font-size="14">{label}</text>')

    # Bar — params
    b1 = bar(X_C1_L, X_C1_R, p, MAX_P)
    svg.append(f'<rect x="{X_C1_L}" y="{y_bar_top}" width="{b1 - X_C1_L:.1f}" '
               f'height="{y_bar_h}" fill="#3a3a3a"/>')
    svg.append(f'<text x="{b1 + 5:.1f}" y="{y_mid + 4}" font-size="12" '
               f'fill="#222">{p:.1f}</text>')

    # Bar — latency
    b2 = bar(X_C2_L, X_C2_R, lat, MAX_L)
    svg.append(f'<rect x="{X_C2_L}" y="{y_bar_top}" width="{b2 - X_C2_L:.1f}" '
               f'height="{y_bar_h}" fill="#3a3a3a"/>')
    svg.append(f'<text x="{b2 + 5:.1f}" y="{y_mid + 4}" font-size="12" '
               f'fill="#222">{lat:.0f}</text>')

    # Bar — FLOPs
    b3 = bar(X_C3_L, X_C3_R, fl, MAX_F)
    svg.append(f'<rect x="{X_C3_L}" y="{y_bar_top}" width="{b3 - X_C3_L:.1f}" '
               f'height="{y_bar_h}" fill="#3a3a3a"/>')
    svg.append(f'<text x="{b3 + 5:.1f}" y="{y_mid + 4}" font-size="12" '
               f'fill="#222">{fl:.1f}</text>')

    # Note text (compute pattern in italic, second line for short hint)
    svg.append(f'<text x="{X_NOTE_L}" y="{y_mid - 1}" font-size="12" '
               f'font-style="italic" fill="#222">{pattern}</text>')
    svg.append(f'<text x="{X_NOTE_L}" y="{y_mid + 13}" font-size="11" '
               f'fill="#666">{note}</text>')

# Range-frame axis lines (thin, only spanning the data range)
y_axis_bottom = M_TOP + ROW_H * N + 4
def axis(x0, x1, maxv, label_max):
    svg.append(f'<line x1="{x0}" y1="{y_axis_bottom}" x2="{x1}" y2="{y_axis_bottom}" '
               f'stroke="#222" stroke-width="0.6"/>')
    svg.append(f'<text x="{x0}" y="{y_axis_bottom + 14}" font-size="10" '
               f'fill="#555">0</text>')
    svg.append(f'<text x="{x1}" y="{y_axis_bottom + 14}" text-anchor="end" '
               f'font-size="10" fill="#555">{label_max}</text>')

axis(X_C1_L, X_C1_R, MAX_P, "50")
axis(X_C2_L, X_C2_R, MAX_L, "580")
axis(X_C3_L, X_C3_R, MAX_F, "8")

# Footer with the punchline
svg.append(f'<text x="40" y="{H - 18}" font-size="12" fill="#444" font-style="italic">'
           f'Read across each row: the same architecture, four ways. '
           f'Mamba pays linearly for sequence length but offers the smallest CPU latency; '
           f'MoE buys 5× FFN capacity at ~2× compute; '
           f'HRM is tiny but deeply sequential, so the GPU will accelerate it less than the parallel attentions.</text>')

svg.append('</svg>')

out_path = "/sessions/vigilant-magical-pascal/mnt/outputs/encoder_comparison.svg"
with open(out_path, "w") as f:
    f.write("\n".join(svg))
print("wrote", out_path)
