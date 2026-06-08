"""
a100/make_seq_class_ids.py
--------------------------
Generate `seq_class_ids` (phoneme labels) LOCALLY from transcriptions, so conditions
C2 (LM) / C4 (oracle) can run on `preprocessed_data.h5` (neural + transcription) WITHOUT the
authors' formatted data.

How: g2p_en(transcription) -> ARPAbet -> 41-class ids (core/phonemes.PHONEMES ordering), with
SIL between words. Writes a new HDF5 with neural + seq_class_ids + transcription + session attr,
in the format core/drift_eval.load_real_sessions expects.

CAVEAT — faithfulness: the trained CTC head was trained on the *authors'* phonemization. g2p_en
may differ slightly, so this is an APPROXIMATE oracle. Validate with --validate_with_model:
it decodes clean trials and reports PER(model_greedy_decode vs this g2p label). If that PER is
near the model's training PER (~low), the inventory + ordering match and C2/C4 are trustworthy.
If it is ~chance (high), the ordering is wrong -> re-obtain the authors' formatted data instead
(restore your training volume, or Dryad + DietCorp's formatCompetitionData). PER-based C0/C1
never need this file.

Usage:
  python a100/make_seq_class_ids.py --in data/preprocessed_data.h5 --out data/sessions_g2p.h5
  python a100/make_seq_class_ids.py --in ... --validate_with_model data/best_model_per.pth \
         --session_stats data/session_stats.json
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.phonemes import arpabet_to_head_id, PHONEMES


def text_to_seq_ids(text, g2p, sil_id_raw):
    """transcription -> raw seq_class_ids (0..40). SIL between words. Returns np.int64 array."""
    ids = []
    words = re.findall(r"[a-zA-Z']+", text.lower())
    for wi, w in enumerate(words):
        for p in g2p(w):
            p = re.sub(r"\d", "", p)            # strip stress
            if not p or not p[0].isalpha():
                continue
            hid = arpabet_to_head_id(p)          # 1..41
            if hid is not None:
                ids.append(hid - 1)              # store raw 0..40 (loader re-adds +1)
        if wi < len(words) - 1 and sil_id_raw is not None:
            ids.append(sil_id_raw)               # SIL between words
    return np.array(ids, dtype=np.int64)


def _read_text(g):
    for tk in ("text", "sentenceText", "transcription"):
        if tk in g:
            rt = g[tk][()]
            if isinstance(rt, np.ndarray):
                rt = rt.astype(np.uint8).tobytes()
            return (rt.rstrip(b"\x00").decode("utf-8", "replace").strip()
                    if isinstance(rt, bytes) else str(rt).strip())
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate_with_model", default=None)
    ap.add_argument("--session_stats", default=None)
    ap.add_argument("--max_trials", type=int, default=None)
    args = ap.parse_args()

    try:
        from g2p_en import G2p
    except Exception as e:
        print(f"g2p_en required: pip install g2p_en  ({e})"); sys.exit(1)
    g2p = G2p()
    sil_raw = (arpabet_to_head_id("SIL") - 1) if arpabet_to_head_id("SIL") else None

    # ── Generate ───────────────────────────────────────────────────────────────
    if args.out:
        with h5py.File(args.inp, "r") as fin, h5py.File(args.out, "w") as fout:
            keys = list(fin.keys())
            if args.max_trials:
                keys = keys[:args.max_trials]
            n_ok = 0
            for k in keys:
                gin = fin[k]
                nkey = next((x for x in ("neural", "input_features") if x in gin), None)
                if nkey is None:
                    continue
                text = _read_text(gin)
                ids = text_to_seq_ids(text, g2p, sil_raw)
                if ids.size == 0:
                    continue
                gout = fout.create_group(k)
                gout.create_dataset("neural", data=gin[nkey][:])
                gout.create_dataset("seq_class_ids", data=ids)
                gout.create_dataset("transcription",
                                    data=np.frombuffer(text.encode("utf-8"), dtype=np.uint8))
                for ak, av in gin.attrs.items():
                    gout.attrs[ak] = av
                n_ok += 1
            print(f"Wrote {n_ok} trials with g2p seq_class_ids -> {args.out}")
            print(f"(inventory: {len(PHONEMES)} phonemes; SIL raw-id={sil_raw})")

    # ── Validate against the trained model (inventory/ordering check) ───────────
    if args.validate_with_model:
        import torch
        from core.model import build_ctc_model
        from core.drift_eval import load_real_sessions, decode_phonemes, per
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_ctc_model(args.validate_with_model, device=device)
        stats = None
        if args.session_stats:
            import json
            stats = json.load(open(args.session_stats))
        # load a few trials WITH g2p labels (from the file we just wrote, or regenerate inline)
        src = args.out or args.inp
        days = load_real_sessions(src, split="val", session_stats=stats, max_per_session=10)
        if not days:
            days = load_real_sessions(src, split="train", session_stats=stats, max_per_session=10)
        logits = lambda x, **kw: model.head(model.encode(x, session_id=kw.get("session_id")))
        pers, n = [], 0
        for sess, trials in list(days.items())[:3]:           # first ~3 days (cleanest)
            for tr in trials:
                ref = tr["ref"]
                if ref is None:
                    continue
                hyp, _ = decode_phonemes(logits, tr["neural"].to(device), tr["session"])
                pers.append(per(hyp, ref)); n += 1
        if n:
            mean = float(np.mean(pers))
            verdict = ("OK: inventory/ordering match the trained head"
                       if mean < 0.6 else "WARNING: high PER -> ordering likely WRONG; re-obtain authors' data")
            print(f"[validate] model-greedy vs g2p-label PER = {mean:.4f} over {n} trials -> {verdict}")
        else:
            print("[validate] no labeled trials available to validate.")


if __name__ == "__main__":
    main()
