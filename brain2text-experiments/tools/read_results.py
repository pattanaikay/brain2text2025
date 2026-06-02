"""
tools/read_results.py
---------------------
Read and display the drift-eval results from any G2/G3 run.

Usage:
    py -3 tools/read_results.py                    # auto-finds the latest run
    py -3 tools/read_results.py G3                 # latest run for a specific expt
    py -3 tools/read_results.py results/runs/G3_adapt_abc123/drift_results.json
"""

import sys, json, os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RUNS = _ROOT / "results" / "runs"


def find_latest(prefix=None):
    if not _RUNS.exists():
        return None
    dirs = [d for d in sorted(_RUNS.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            if d.is_dir() and (prefix is None or d.name.upper().startswith(prefix.upper()))]
    for d in dirs:
        j = d / "drift_results.json"
        if j.exists():
            return j
    return None


def print_table(data: dict, path: Path):
    print(f"\n{'='*65}")
    print(f"  Results: {path.parent.name}")
    print(f"{'='*65}")

    # Header
    print(f"\n{'N':>4}  {'PER@day0':>9}  {'PER@last':>9}  {'delta':>8}  {'wake_ms':>9}  {'cons_ms':>9}")
    print(f"{'─'*4}  {'─'*9}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*9}")

    ns = sorted(data["summary"].keys(), key=int)
    for N in ns:
        s   = data["summary"][N]
        w   = data["wake_latency_ms"].get(N)
        c   = data["consolidate_ms"].get(N)
        p0  = s["per_first"]
        pl  = s["per_last"]
        delta = s["per_delta"]

        def _f(v, dec=4):
            return f"{v:.{dec}f}" if isinstance(v, (int, float)) else "   n/a"

        print(f"{N:>4}  {_f(p0):>9}  {_f(pl):>9}  {_f(delta):>8}  {_f(w,2):>9}  {_f(c,2):>9}")

    print()

    # Interpretation
    pers_last = [(int(N), data["summary"][N]["per_last"]) for N in ns
                 if data["summary"][N]["per_last"] is not None]
    wakes = [(int(N), data["wake_latency_ms"][N]) for N in ns
             if data["wake_latency_ms"].get(N) is not None]

    if len(pers_last) >= 2:
        pls = [p for _, p in sorted(pers_last)]
        best_n, best_p = min(pers_last, key=lambda x: x[1])
        no_adapt_p = dict(pers_last).get(0)

        print("  INTERPRETATION:")

        if no_adapt_p is not None and best_p < no_adapt_p:
            gain = (no_adapt_p - best_p) / no_adapt_p * 100
            print(f"  - Adaptation works: best PER@last is {best_p:.4f} at N={best_n}")
            print(f"    vs no-adapt baseline {no_adapt_p:.4f}  ({gain:.1f}% relative reduction)")
        elif no_adapt_p is not None:
            print(f"  WARNING: adaptation did not beat no-adapt baseline ({no_adapt_p:.4f}).")
            print(f"  Likely cause: pseudo-label collapse (checkpoint not trained enough,")
            print(f"  or confidence threshold too low). Check confidence values in JSON.")

        # Check N-monotonicity
        adapted = [(n, p) for n, p in sorted(pers_last) if n > 0]
        if len(adapted) >= 2:
            mono = all(adapted[i][1] >= adapted[i+1][1] for i in range(len(adapted)-1))
            if mono:
                print(f"  - N-monotonicity: YES. PER falls as N increases (H_main supported)")
            else:
                best_idx = adapted.index(min(adapted, key=lambda x: x[1]))
                print(f"  - N-monotonicity: PARTIAL. Best N={adapted[best_idx][0]},")
                print(f"    then instability (matches Sleep paper stability limit)")
                print(f"    Recommended: use N={adapted[best_idx][0]} for cloud run")

        # Wake latency flatness
        if len(wakes) >= 2:
            wake_vals = [w for _, w in wakes]
            spread = (max(wake_vals) - min(wake_vals)) / max(wake_vals) * 100
            if spread < 20:
                print(f"  - Wake latency: FLAT across N (spread {spread:.1f}%) -- clinical constraint met")
            else:
                print(f"  WARNING: Wake latency varies {spread:.1f}% across N -- investigate")

    print()
    print(f"  Full data: {path}")
    print()

    # Day-by-day curve for each N
    print("  PER per day (all N):")
    print(f"  {'Day':<8}", end="")
    for N in ns:
        print(f"  N={N:<6}", end="")
    print()
    n_days = max(len(data["by_n"][N]) for N in ns)
    for d in range(n_days):
        row = data["by_n"][ns[0]]
        day_name = row[d]["day"] if d < len(row) else f"d{d}"
        print(f"  {day_name:<8}", end="")
        for N in ns:
            curve = data["by_n"][N]
            per = curve[d]["per"] if d < len(curve) else None
            val = f"{per:.4f}" if per is not None else "  n/a "
            print(f"  {val:<8}", end="")
        print()
    print()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    path = None

    if arg and os.path.isfile(arg):
        path = Path(arg)
    elif arg:
        path = find_latest(prefix=arg)
    else:
        path = find_latest()

    if path is None:
        print("No drift_results.json found.")
        print("Run: py -3 run.py --expt G3 --profile toy --adapt")
        print("     py -3 run.py --expt G2 --profile toy --adapt")
        sys.exit(1)

    data = json.loads(path.read_text())
    print_table(data, path)


if __name__ == "__main__":
    main()
