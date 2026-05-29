"""
results/leaderboard.py
----------------------
SQLite-backed Pareto-frontier leaderboard.

Every completed run is recorded. run.py calls record_run() at the end.
promote() prints the current Pareto frontier (WER vs GPU-hours).

Usage:
    python results/leaderboard.py --list              # show all runs
    python results/leaderboard.py --frontier          # Pareto frontier only
    python results/leaderboard.py --promote B1 toy    # check if B1 toy qualifies for A100
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

_DB = Path(__file__).parent / "leaderboard.sqlite"


def _conn():
    db = sqlite3.connect(str(_DB))
    db.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            expt_id     TEXT    NOT NULL,
            profile     TEXT    NOT NULL,
            spec_hash   TEXT    NOT NULL,
            best_wer    REAL,
            wer_at_ep10 REAL,
            slope       REAL,
            gpu_hours   REAL,
            run_dir     TEXT,
            ts          INTEGER NOT NULL,
            notes       TEXT
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS smoke_passes (
            expt_id   TEXT NOT NULL,
            profile   TEXT NOT NULL,
            spec_hash TEXT NOT NULL,
            ts        INTEGER NOT NULL
        )
    """)
    db.commit()
    return db


def record_run(
    expt_id:     str,
    profile:     str,
    spec_hash:   str,
    best_wer:    float  = None,
    wer_at_ep10: float  = None,
    slope:       float  = None,
    gpu_hours:   float  = None,
    run_dir:     str    = None,
    notes:       str    = None,
):
    db = _conn()
    db.execute("""
        INSERT INTO runs (expt_id, profile, spec_hash, best_wer, wer_at_ep10,
                          slope, gpu_hours, run_dir, ts, notes)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (expt_id, profile, spec_hash, best_wer, wer_at_ep10,
          slope, gpu_hours, run_dir, int(time.time()), notes))
    db.commit()
    db.close()
    print(f"[leaderboard] Recorded {expt_id}/{profile} → best_wer={best_wer}")

    # Auto-record smoke pass if WER is sensible
    if best_wer is not None and best_wer < 1.0:
        record_smoke_pass(expt_id, profile, spec_hash)


def record_smoke_pass(expt_id: str, profile: str, spec_hash: str):
    db = _conn()
    db.execute("""
        INSERT INTO smoke_passes (expt_id, profile, spec_hash, ts)
        VALUES (?,?,?,?)
    """, (expt_id, profile, spec_hash, int(time.time())))
    db.commit()
    db.close()


def toy_passed_recently(expt_id: str, days: int = 7) -> bool:
    """Check if expt_id has a toy PASSED in the last `days` days."""
    cutoff = int(time.time()) - days * 86400
    db = _conn()
    row = db.execute("""
        SELECT COUNT(*) FROM smoke_passes
        WHERE expt_id=? AND profile='toy' AND ts > ?
    """, (expt_id, cutoff)).fetchone()
    db.close()
    return row[0] > 0


def list_runs(profile=None, sort_by="best_wer") -> list[dict]:
    db = _conn()
    q  = "SELECT * FROM runs"
    p  = []
    if profile:
        q += " WHERE profile=?"; p.append(profile)
    q += f" ORDER BY {sort_by} ASC NULLS LAST"
    rows = db.execute(q, p).fetchall()
    cols = [d[0] for d in db.execute("SELECT * FROM runs LIMIT 0").description]
    db.close()
    return [dict(zip(cols, r)) for r in rows]


def pareto_frontier() -> list[dict]:
    """
    Pareto frontier: runs that are not dominated on both (best_wer, gpu_hours).
    Returns sorted by best_wer ascending.
    """
    runs = [r for r in list_runs() if r["best_wer"] is not None]
    frontier = []
    for r in sorted(runs, key=lambda x: x["best_wer"]):
        dominated = any(
            f["best_wer"] <= r["best_wer"] and
            (f["gpu_hours"] or 0) <= (r["gpu_hours"] or 0)
            for f in frontier
        )
        if not dominated:
            frontier.append(r)
    return frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list",     action="store_true")
    parser.add_argument("--frontier", action="store_true")
    parser.add_argument("--promote",  nargs=2, metavar=("EXPT_ID", "PROFILE"))
    args = parser.parse_args()

    if args.list:
        runs = list_runs()
        print(f"{'ID':<6} {'expt':<15} {'profile':<8} {'best_wer':<10} {'wer@10':<10} {'slope':<10}")
        print("-" * 65)
        for r in runs:
            print(f"{r['id']:<6} {r['expt_id']:<15} {r['profile']:<8} "
                  f"{r['best_wer'] or '?':<10.4f} "
                  f"{r['wer_at_ep10'] or '?':<10.4f} "
                  f"{r['slope'] or '?':<10.6f}")
        return

    if args.frontier:
        runs = pareto_frontier()
        print("Pareto frontier (WER vs GPU-hours):")
        for r in runs:
            print(f"  {r['expt_id']}/{r['profile']}  WER={r['best_wer']:.4f}  "
                  f"gpu_hours={r['gpu_hours'] or '?'}")
        return

    if args.promote:
        expt_id, profile = args.promote
        ok = toy_passed_recently(expt_id, days=7)
        if ok:
            print(f"✓ {expt_id} toy run PASSED recently — eligible for full/A100 run")
        else:
            print(f"✗ {expt_id} toy run NOT found in last 7 days — run toy first")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
