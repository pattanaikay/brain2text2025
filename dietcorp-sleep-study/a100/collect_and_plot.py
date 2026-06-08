"""
a100/collect_and_plot.py
------------------------
Read results/study_results.json and produce the headline figure + summary tables.

Figure: PER-vs-held-out-day, one panel per condition, one curve per N. This is the artifact
for the defense: H_main is visible iff, in C2/C3, the higher-N curves sit below N=1 at later
days while wake latency (printed) stays flat.

Usage: python a100/collect_and_plot.py results/
"""

import json
import sys
from pathlib import Path


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    data = json.loads((results_dir / "study_results.json").read_text())

    # ── Summary tables ─────────────────────────────────────────────────────────
    print("\n=== Summary: PER@last-day and latency (per condition x N) ===")
    for cond in sorted(data.keys()):
        print(f"\nCondition {cond}")
        print(f"  {'N':>3} {'PER@day0':>9} {'PER@last':>9} {'WER@last':>9} {'wake_ms':>9} {'cons_ms':>9}")
        by_n = data[cond]["by_n"]
        for N in sorted(by_n, key=int):
            curve = by_n[N]
            pers = [c["per"] for c in curve if c["per"] is not None]
            p0, pl = (pers[0], pers[-1]) if pers else (None, None)
            wers = [c.get("wer") for c in curve if c.get("wer") is not None]
            wl = wers[-1] if wers else None
            wake = data[cond]["wake_latency_ms"].get(N)
            cons = data[cond]["consolidate_ms"].get(N)
            def f(x, d=4): return f"{x:.{d}f}" if isinstance(x, (int, float)) else "   n/a"
            print(f"  {N:>3} {f(p0):>9} {f(pl):>9} {f(wl):>9} {f(wake,2):>9} {f(cons,2):>9}")

    # ── Read-off ───────────────────────────────────────────────────────────────
    print("\n=== H_main read-off ===")
    for cond in ("C2", "C3"):
        if cond not in data:
            continue
        by_n = data[cond]["by_n"]
        lasts = {int(N): ([c["per"] for c in by_n[N]][-1]) for N in by_n}
        if 1 in lasts and len(lasts) > 1:
            best_n = min(lasts, key=lasts.get)
            verdict = ("supported" if best_n > 1 and lasts[best_n] < lasts[1] else "not supported")
            print(f"  {cond}: best N={best_n} (PER@last={lasts[best_n]:.4f}) vs N=1 "
                  f"({lasts[1]:.4f}) -> H_main {verdict}")
    if "C2" in data and "C3" in data:
        c2 = min((([c['per'] for c in data['C2']['by_n'][N]][-1]) for N in data['C2']['by_n']), default=None)
        c3 = min((([c['per'] for c in data['C3']['by_n'][N]][-1]) for N in data['C3']['by_n']), default=None)
        if c2 is not None and c3 is not None:
            print(f"  Memory contribution: best C3 PER@last={c3:.4f} vs best C2={c2:.4f} "
                  f"-> memory {'helps' if c3 < c2 else 'does not help'}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        conds = sorted(data.keys())
        fig, axes = plt.subplots(1, len(conds), figsize=(4 * len(conds), 4), sharey=True, squeeze=False)
        for ax, cond in zip(axes[0], conds):
            by_n = data[cond]["by_n"]
            for N in sorted(by_n, key=int):
                curve = by_n[N]
                xs = list(range(len(curve)))
                ys = [c["per"] for c in curve]
                ax.plot(xs, ys, marker="o", label=f"N={N}")
            ax.set_title(cond); ax.set_xlabel("held-out day"); ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0][0].set_ylabel("PER")
        fig.suptitle("Sleep consolidation under drift — PER vs day (per N)")
        fig.tight_layout()
        out = results_dir / "wer_vs_day.png"
        fig.savefig(out, dpi=130)
        print(f"\nSaved figure -> {out}")
    except Exception as e:
        print(f"\n(plot skipped: {e})")


if __name__ == "__main__":
    main()
