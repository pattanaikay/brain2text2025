"""
a100/build_lm.py
----------------
Builds the language resources from the dataset transcriptions / seq_class_ids:
  1. phoneme n-gram LM   -> pseudo-label refinement (conditions C2/C3)
  2. word lexicon + unigram -> word-level WER decoding (a100/collect, run_study --word_decode)

Usage:
  python a100/build_lm.py --data data/sessions --out_dir data/lm --order 3
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.lm_refine import PhonemeNGramLM
from core.phonemes import arpabet_to_head_id
from core.wer_decode import Lexicon


def collect_sequences(data, split="train"):
    paths = []
    if os.path.isdir(data):
        for root, _, files in os.walk(data):
            for fn in files:
                if fn.endswith((".h5", ".hdf5")):
                    paths.append(os.path.join(root, fn))
    else:
        paths = [data]
    seqs = []
    for p in paths:
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            if any("data_train" in k or "data_val" in k for k in keys):
                keys = [k for k in keys if f"data_{split}" in k]
            for k in keys:
                g = f[k]
                for pk in ("phonemes", "seq_class_ids", "phonemeLabels"):
                    if pk in g:
                        ids = [int(x) + 1 for x in g[pk][:]]   # +1 to match training (blank=0)
                        if ids:
                            seqs.append(ids)
                        break
    return seqs


def collect_texts(data, split="train"):
    paths = []
    if os.path.isdir(data):
        for root, _, files in os.walk(data):
            for fn in files:
                if fn.endswith((".h5", ".hdf5")):
                    paths.append(os.path.join(root, fn))
    else:
        paths = [data]
    texts = []
    for p in paths:
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            if any("data_train" in k or "data_val" in k for k in keys):
                keys = [k for k in keys if f"data_{split}" in k]
            for k in keys:
                g = f[k]
                for tk in ("text", "sentenceText", "transcription"):
                    if tk in g:
                        rt = g[tk][()]
                        if isinstance(rt, np.ndarray):
                            rt = rt.astype(np.uint8).tobytes()
                        t = (rt.rstrip(b"\x00").decode("utf-8", "replace").strip()
                             if isinstance(rt, bytes) else str(rt).strip())
                        if t:
                            texts.append(t)
                        break
    return texts


def build_lexicon(texts, out_path):
    """word -> phoneme-id pronunciation (via g2p_en) + word unigram from the corpus."""
    try:
        from g2p_en import G2p
    except Exception as e:
        print(f"g2p_en not available ({e}); skipping lexicon. `pip install g2p_en` to enable WER.")
        return
    g2p = G2p()
    counts = Counter()
    for t in texts:
        for w in re.findall(r"[a-zA-Z']+", t.lower()):
            counts[w] += 1
    total = sum(counts.values()) or 1
    prons, unigram = {}, {}
    for w, c in counts.items():
        arpa = [re.sub(r"\d", "", p) for p in g2p(w) if p.strip() and p[0].isalpha()]
        ids = [arpabet_to_head_id(a) for a in arpa]
        ids = [i for i in ids if i is not None]
        if ids:
            prons[w] = ids
            unigram[w] = float(np.log(c / total))
    Lexicon(prons, unigram).save(out_path)
    print(f"Saved lexicon -> {out_path} ({len(prons)} words)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default="data/lm")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. phoneme n-gram (pseudo-label refinement)
    seqs = collect_sequences(args.data, split=args.split)
    if seqs:
        print(f"Training {args.order}-gram on {len(seqs)} phoneme sequences "
              f"({sum(len(s) for s in seqs)} tokens)...")
        lm = PhonemeNGramLM(order=args.order).fit(seqs)
        lm.save(os.path.join(args.out_dir, f"phoneme_{args.order}gram.json"))
        print(f"Saved phoneme LM (vocab={len(lm.vocab)})")
    else:
        print("No seq_class_ids found -> no phoneme LM (C2/C3 will fall back to self-label).")

    # 2. lexicon + word unigram (WER decoding)
    texts = collect_texts(args.data, split=args.split)
    if texts:
        print(f"Building lexicon from {len(texts)} transcriptions...")
        build_lexicon(texts, os.path.join(args.out_dir, "lexicon.json"))
    else:
        print("No transcriptions found -> no lexicon (WER decode unavailable).")


if __name__ == "__main__":
    main()
