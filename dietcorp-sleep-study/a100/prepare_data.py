"""
a100/prepare_data.py
--------------------
Verify the session data is in the format the study needs and print a session inventory.

The headline run needs per-trial neural + `seq_class_ids` (phoneme labels) + a session id —
the format produced by DietCorp's notebooks/formatCompetitionData.ipynb from the Dryad data
(see a100/SYNC.md). This script does NOT download/format; it validates what is present and
reports which conditions are runnable (oracle/LM need seq_class_ids).
"""

import argparse
import os
import sys
from collections import Counter

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir of *.hdf5 or a single .h5")
    args = ap.parse_args()

    paths = []
    if os.path.isdir(args.data):
        for root, _, files in os.walk(args.data):
            for fn in files:
                if fn.endswith((".h5", ".hdf5")):
                    paths.append(os.path.join(root, fn))
    else:
        paths = [args.data]

    if not paths:
        print(f"No .h5/.hdf5 found under {args.data}"); sys.exit(1)

    sessions = Counter()
    have_phonemes = have_text = total = 0
    import re
    DATE = re.compile(r"(t1\d[._]\d{4}[._]\d{2}[._]\d{2})")
    for p in paths:
        with h5py.File(p, "r") as f:
            for k in f.keys():
                g = f[k]
                total += 1
                sess = g.attrs.get("session", None)
                if sess is None:
                    m = DATE.search(k); sess = m.group(1) if m else "unknown"
                sessions[str(sess)] += 1
                if any(x in g for x in ("phonemes", "seq_class_ids", "phonemeLabels")):
                    have_phonemes += 1
                if any(x in g for x in ("text", "sentenceText", "transcription")):
                    have_text += 1

    print(f"Files: {len(paths)} | trials: {total} | sessions: {len(sessions)}")
    print(f"Trials with phoneme labels (seq_class_ids): {have_phonemes}/{total}")
    print(f"Trials with text: {have_text}/{total}")
    print("Sessions (chronological):")
    for s in sorted(sessions):
        print(f"  {s}: {sessions[s]} trials")
    print()
    if have_phonemes == 0:
        print("WARNING: no phoneme labels found -> C2(LM)/C4(oracle) need them. "
              "Format the Dryad data with seq_class_ids (see a100/SYNC.md). "
              "C0/C1 (self-reference PER) still run.")
    else:
        print("OK: phoneme labels present -> all conditions C0-C4 runnable.")


if __name__ == "__main__":
    main()
