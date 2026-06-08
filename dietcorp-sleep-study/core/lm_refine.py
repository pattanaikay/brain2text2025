"""
core/lm_refine.py
-----------------
n-gram pseudo-label refinement — the fix for N>1 collapse.

Raw greedy CTC decodes on drifted signals are noisy; using them as gradient targets makes
each consolidation step follow a slightly wrong direction, and N steps compound the error.
DietCorp avoids this with a 3-gram LM. Here we provide:

  - PhonemeNGramLM: a pure-Python n-gram (add-k + stupid backoff) over phoneme-ID sequences,
    trained from the dataset's `seq_class_ids`. No external deps; portable to the A100.
    (A KenLM .arpa can be loaded via `from_arpa` when `kenlm` is installed — optional.)
  - ctc_beam_search(log_probs, lm, ...): CTC prefix beam search with LM shallow fusion,
    returning a denoised phoneme-ID sequence to use as the pseudo-label.

Tokens: 0 = CTC blank; 1..V-1 = phonemes. The LM is defined over the phoneme tokens (>=1).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import torch
import torch.nn.functional as F


class PhonemeNGramLM:
    def __init__(self, order: int = 3, k: float = 0.1):
        self.order = order
        self.k = k                                  # add-k smoothing
        self.ngrams = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.vocab: set[int] = set()

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, sequences: list[list[int]]):
        """sequences: list of phoneme-ID lists (>=1; no blanks)."""
        for seq in sequences:
            self.vocab.update(seq)
            padded = [-1] * (self.order - 1) + list(seq) + [-2]   # BOS=-1, EOS=-2
            for n in range(1, self.order + 1):
                tbl = self.ngrams[n - 1]
                for i in range(len(padded) - n + 1):
                    ctx = tuple(padded[i:i + n - 1])
                    tok = padded[i + n - 1]
                    tbl[ctx][tok] += 1
        self.vocab.discard(-1); self.vocab.discard(-2)
        return self

    @property
    def V(self) -> int:
        return max(2, len(self.vocab) + 1)

    # ── Scoring ───────────────────────────────────────────────────────────────
    def logprob(self, token: int, context: tuple) -> float:
        """log P(token | context) with stupid backoff over decreasing context length."""
        for n in range(min(self.order, len(context) + 1), 0, -1):
            ctx = tuple(context[-(n - 1):]) if n > 1 else ()
            tbl = self.ngrams[n - 1]
            if ctx in tbl:
                counts = tbl[ctx]
                total = sum(counts.values())
                c = counts.get(token, 0)
                if c > 0:
                    return math.log((c + self.k) / (total + self.k * self.V))
            # backoff with a small penalty
        # unseen everywhere → uniform
        return math.log(1.0 / self.V) - 1.0

    # ── Persistence ────────────────────────────────────────────────────────────
    def save(self, path: str):
        obj = {"order": self.order, "k": self.k, "vocab": sorted(self.vocab),
               "ngrams": [{",".join(map(str, ctx)): dict(toks) for ctx, toks in tbl.items()}
                          for tbl in self.ngrams]}
        with open(path, "w") as f:
            json.dump(obj, f)

    @classmethod
    def load(cls, path: str) -> "PhonemeNGramLM":
        with open(path) as f:
            obj = json.load(f)
        lm = cls(order=obj["order"], k=obj["k"])
        lm.vocab = set(obj["vocab"])
        for n, tbl in enumerate(obj["ngrams"]):
            for ctx_s, toks in tbl.items():
                ctx = tuple(int(x) for x in ctx_s.split(",")) if ctx_s else ()
                lm.ngrams[n][ctx] = defaultdict(int, {int(t): c for t, c in toks.items()})
        return lm


def ctc_beam_search(log_probs: torch.Tensor, lm: PhonemeNGramLM | None = None,
                    beam_size: int = 8, lm_weight: float = 0.5,
                    blank: int = 0, max_frames: int = 200) -> list[int]:
    """
    CTC prefix beam search with optional n-gram LM shallow fusion.
    log_probs: (T, V) log-softmax for ONE trial. Returns a phoneme-ID list (collapsed, no blank).
    """
    if log_probs.dim() == 3:
        log_probs = log_probs[0]
    T, V = log_probs.shape
    lp = log_probs.detach().cpu()
    if T > max_frames:                                  # subsample very long trials for speed
        idx = torch.linspace(0, T - 1, max_frames).long()
        lp = lp[idx]; T = max_frames

    NEG = -1e30
    # beam: prefix(tuple) -> (p_blank, p_nonblank) in log space
    beams = {(): (0.0, NEG)}

    def _lm_score(prefix, tok):
        if lm is None or lm_weight == 0.0:
            return 0.0
        return lm_weight * lm.logprob(tok, prefix)

    for t in range(T):
        row = lp[t]
        # candidate tokens: top-k by emission prob (keeps it fast)
        topk = torch.topk(row, min(beam_size + 1, V)).indices.tolist()
        if blank not in topk:
            topk.append(blank)
        next_beams: dict[tuple, list] = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            ptot = _logsumexp(pb, pnb)
            for c in topk:
                lc = float(row[c])
                if c == blank:
                    nb, nnb = next_beams[prefix]
                    next_beams[prefix] = [_logsumexp(nb, ptot + lc), nnb]
                else:
                    if prefix and c == prefix[-1]:
                        # repeat: extend only from blank path; stay-same from nonblank
                        nb, nnb = next_beams[prefix]
                        next_beams[prefix] = [nb, _logsumexp(nnb, pnb + lc)]
                        newp = prefix + (c,)
                        nb2, nnb2 = next_beams[newp]
                        next_beams[newp] = [nb2, _logsumexp(nnb2, pb + lc + _lm_score(prefix, c))]
                    else:
                        newp = prefix + (c,)
                        nb2, nnb2 = next_beams[newp]
                        next_beams[newp] = [nb2, _logsumexp(nnb2, ptot + lc + _lm_score(prefix, c))]
        # prune
        scored = sorted(next_beams.items(),
                        key=lambda kv: _logsumexp(kv[1][0], kv[1][1]), reverse=True)
        beams = {p: tuple(v) for p, v in scored[:beam_size]}

    best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
    return list(best)


def _logsumexp(a: float, b: float) -> float:
    if a == -1e30: return b
    if b == -1e30: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
