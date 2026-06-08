"""
core/wer_decode.py
------------------
Phoneme-sequence -> words -> WER (word-level), the metric comparable to DietCorp's 12.17%.

A lexicon noisy-channel decoder: segment the emitted phoneme-id stream into words by matching
against a pronunciation lexicon (word -> phoneme-id sequence), minimizing
    sum_words [ edit_distance(segment, pron(word)) + lm_weight * (-log P_unigram(word)) ]
with free skipping of boundary tokens (SIL/SP). Exact pronunciations cost 0, so decoding the
GROUND-TRUTH seq_class_ids reproduces the transcription (WER ~ 0) — that is the self-check
that validates the phoneme inventory (see run_study --validate_mapping).

Lexicon + unigram are built by a100/build_lm.py from the transcriptions (g2p_en pronunciations).
For a fully faithful word LM you can instead use DietCorp's own n-gram decoder (see README).
"""

from __future__ import annotations

import json
import math

import jiwer

from core.phonemes import BOUNDARY, arpabet_to_head_id


def _edit(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Lexicon:
    """word -> pronunciation as a list of head-ids (1..41)."""

    def __init__(self, prons: dict[str, list[int]], unigram: dict[str, float] | None = None):
        self.prons = prons
        self.unigram = unigram or {}
        self.buckets: dict[int, list[tuple[str, list[int]]]] = {}
        for w, p in prons.items():
            self.buckets.setdefault(len(p), []).append((w, p))
        self.boundary_ids = {arpabet_to_head_id(b) for b in BOUNDARY} - {None}

    def unigram_penalty(self, word: str) -> float:
        lp = self.unigram.get(word)
        return -lp if lp is not None else 6.0   # mild penalty for OOV/rare

    def save(self, path):
        json.dump({"prons": self.prons, "unigram": self.unigram}, open(path, "w"))

    @classmethod
    def load(cls, path):
        obj = json.load(open(path))
        return cls({w: list(p) for w, p in obj["prons"].items()}, obj.get("unigram"))


def decode_words(phoneme_head_ids, lexicon: Lexicon, lm_weight: float = 1.0,
                 len_tol: int = 1) -> list[str]:
    """DP segmentation of a phoneme-id list into the lowest-cost word sequence."""
    P = [int(x) for x in (phoneme_head_ids.tolist() if hasattr(phoneme_head_ids, "tolist")
                          else phoneme_head_ids)]
    T = len(P)
    INF = 1e18
    best = [INF] * (T + 1); best[0] = 0.0
    back = [None] * (T + 1)
    lengths = sorted(lexicon.buckets.keys())
    for i in range(1, T + 1):
        # free-skip a boundary token
        if P[i - 1] in lexicon.boundary_ids and best[i - 1] + 0.0 < best[i]:
            best[i] = best[i - 1]; back[i] = (i - 1, None)
        for L in lengths:
            for j in range(max(0, i - L - len_tol), max(0, i - L + len_tol) + 1):
                if j >= i or best[j] == INF:
                    continue
                seg = P[j:i]
                for w, pron in lexicon.buckets[L]:
                    c = _edit(seg, pron) + lm_weight * lexicon.unigram_penalty(w)
                    if best[j] + c < best[i]:
                        best[i] = best[j] + c; back[i] = (j, w)
    # reconstruct
    words, i = [], T
    guard = 0
    while i > 0 and back[i] is not None and guard < 10000:
        j, w = back[i]
        if w is not None:
            words.append(w)
        i = j; guard += 1
    return list(reversed(words))


def compute_wer(pred_words: list[str], ref_text: str) -> float:
    ref = ref_text.strip().lower()
    hyp = " ".join(pred_words).strip().lower()
    if not ref:
        return 0.0 if not hyp else 1.0
    return jiwer.wer(ref, hyp)
