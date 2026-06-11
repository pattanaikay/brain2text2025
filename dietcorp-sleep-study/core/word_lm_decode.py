"""
core/word_lm_decode.py
----------------------
Canonical word-level decoding: phoneme-id stream -> words with a WORD n-gram LM (not just the
unigram penalty of core/wer_decode.decode_words). This is the citable WER decoder — a lexicon
beam search with shallow word-LM fusion, comparable to the Brain-to-Text n-gram baseline.

  cost(word_seq) = sum_w [ edit_distance(segment, pron(w))
                           + lm_weight      * (-log P_unigram(w))
                           + word_lm_weight * (-log P_ngram(w | preceding words)) ]

The word LM is trained from the dataset transcriptions by a100/build_lm.py. Matching DietCorp's
exact headline (12.17%) additionally needs their KenLM + LLM rescoring — that is a stretch goal;
this n-gram word-LM decode is the workshop-level canonical number. Falls back to the lexicon-DP
decoder if no path reaches the end.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

from core.wer_decode import Lexicon, _edit, decode_words


class WordNGramLM:
    """add-k + stupid-backoff n-gram over word tokens."""

    def __init__(self, order: int = 3, k: float = 0.1):
        self.order = order
        self.k = k
        self.ngrams = [defaultdict(lambda: defaultdict(int)) for _ in range(order)]
        self.vocab: set[str] = set()

    def fit(self, sentences: list[list[str]]):
        for toks in sentences:
            self.vocab.update(toks)
            padded = ["<s>"] * (self.order - 1) + list(toks) + ["</s>"]
            for n in range(1, self.order + 1):
                tbl = self.ngrams[n - 1]
                for i in range(len(padded) - n + 1):
                    ctx = tuple(padded[i:i + n - 1])
                    tbl[ctx][padded[i + n - 1]] += 1
        self.vocab.discard("<s>"); self.vocab.discard("</s>")
        return self

    @property
    def V(self) -> int:
        return max(2, len(self.vocab) + 1)

    def logprob(self, token: str, context: tuple) -> float:
        for n in range(min(self.order, len(context) + 1), 0, -1):
            ctx = tuple(context[-(n - 1):]) if n > 1 else ()
            tbl = self.ngrams[n - 1]
            if ctx in tbl:
                counts = tbl[ctx]
                total = sum(counts.values())
                c = counts.get(token, 0)
                if c > 0:
                    return math.log((c + self.k) / (total + self.k * self.V))
        return math.log(1.0 / self.V) - 1.0

    def save(self, path: str):
        obj = {"order": self.order, "k": self.k, "vocab": sorted(self.vocab),
               "ngrams": [{"\t".join(ctx): dict(toks) for ctx, toks in tbl.items()}
                          for tbl in self.ngrams]}
        json.dump(obj, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "WordNGramLM":
        obj = json.load(open(path))
        lm = cls(order=obj["order"], k=obj["k"])
        lm.vocab = set(obj["vocab"])
        for n, tbl in enumerate(obj["ngrams"]):
            for ctx_s, toks in tbl.items():
                ctx = tuple(ctx_s.split("\t")) if ctx_s else ()
                lm.ngrams[n][ctx] = defaultdict(int, {t: int(c) for t, c in toks.items()})
        return lm


def _first_phoneme_index(lex: Lexicon):
    """word candidates keyed by the first head-id of their pronunciation (cached on the lexicon)."""
    idx = getattr(lex, "_fp_index", None)
    if idx is None:
        idx = {}
        for w, pron in lex.prons.items():
            if pron:
                idx.setdefault(int(pron[0]), []).append((w, [int(p) for p in pron], len(pron)))
        lex._fp_index = idx
    return idx


def decode_words_lm(phoneme_head_ids, lexicon: Lexicon, word_lm: WordNGramLM | None = None,
                    lm_weight: float = 1.0, word_lm_weight: float = 1.0,
                    len_tol: int = 1, beam: int = 16) -> list[str]:
    """Position-synchronous lexicon beam search with shallow word-LM fusion."""
    P = [int(x) for x in (phoneme_head_ids.tolist() if hasattr(phoneme_head_ids, "tolist")
                          else phoneme_head_ids)]
    T = len(P)
    if T == 0:
        return []
    idx = _first_phoneme_index(lexicon)
    beams: dict[int, list[tuple[float, tuple]]] = {0: [(0.0, ())]}
    for i in range(T):
        if i not in beams:
            continue
        hyps = sorted(beams[i], key=lambda h: h[0])[:beam]
        beams[i] = hyps
        for score, words in hyps:
            if P[i] in lexicon.boundary_ids:                    # free-skip SIL/SP
                beams.setdefault(i + 1, []).append((score, words))
            for w, pron, L in idx.get(P[i], []):
                for end in range(max(i + 1, i + L - len_tol), min(T, i + L + len_tol) + 1):
                    seg = P[i:end]
                    if not seg:
                        continue
                    cost = _edit(seg, pron) + lm_weight * lexicon.unigram_penalty(w)
                    if word_lm is not None and word_lm_weight:
                        cost += word_lm_weight * (-word_lm.logprob(w, words))
                    beams.setdefault(end, []).append((score + cost, words + (w,)))
    final = sorted(beams.get(T, []), key=lambda h: h[0])
    if final:
        return list(final[0][1])
    return decode_words(phoneme_head_ids, lexicon, lm_weight=lm_weight, len_tol=len_tol)
