"""
core/drift_eval.py
------------------
Real chronological-session drift loading + PER metric.

Sessions are dated in the trial keys (`t15.YYYY.MM.DD_..._trial_NNNN`) or in the `session`
attr, so ordering them chronologically gives a genuine day-to-day drift sequence — no
synthetic perturbation needed. Neural preprocessing replicates the training dataloader
(per-session z-score via session_stats + Gaussian smoothing), so the trained checkpoint
decodes correctly.
"""

from __future__ import annotations

import re
from collections import OrderedDict

import numpy as np
import torch

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter1d = None


# ── PER ──────────────────────────────────────────────────────────────────────

def levenshtein(a, b) -> int:
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


def per(hyp: torch.Tensor, ref: torch.Tensor) -> float:
    ref_list = ref.tolist() if hasattr(ref, "tolist") else list(ref)
    if not ref_list:
        return 1.0 if (hasattr(hyp, "numel") and hyp.numel()) else 0.0
    hyp_list = hyp.tolist() if hasattr(hyp, "tolist") else list(hyp)
    return levenshtein(hyp_list, ref_list) / len(ref_list)


# ── Session parsing ──────────────────────────────────────────────────────────

_DATE = re.compile(r"(t1\d[._]\d{4}[._]\d{2}[._]\d{2})")

def parse_session(trial_key: str, attr_session=None) -> str:
    if attr_session:
        return str(attr_session)
    m = _DATE.search(trial_key)
    return m.group(1).replace("_", ".") if m else "unknown"


# ── Real-session loader ──────────────────────────────────────────────────────

def load_real_sessions(h5_path, split="val", session_stats=None, sigma=1.5,
                       max_per_session=None):
    """
    Returns OrderedDict[session_date -> list of {neural(T,C), ref(L,)|None, text, session}],
    chronologically ordered. `split` filters trial keys containing 'data_val'/'data_train'
    when present (consolidated single-file format); ignored for per-session files.
    """
    import h5py
    by_session = OrderedDict()
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if split and any("data_train" in k or "data_val" in k for k in keys):
            keys = [k for k in keys if f"data_{split}" in k]
        for k in keys:
            g = f[k]
            attr_sess = g.attrs.get("session", None)
            session = parse_session(k, attr_sess)
            # neural
            nkey = next((x for x in ("neural", "input_features", "tx1", "spikePow") if x in g), None)
            if nkey is None:
                continue
            neural = g[nkey][:].astype(np.float32)
            # optional phoneme ref (oracle); +1 to match training (blank=0)
            ref = None
            for pk in ("phonemes", "seq_class_ids", "phonemeLabels"):
                if pk in g:
                    ref = torch.tensor(g[pk][:].astype(np.int64) + 1, dtype=torch.long)
                    break
            # text
            text = ""
            for tk in ("text", "sentenceText", "transcription"):
                if tk in g:
                    rt = g[tk][()]
                    if isinstance(rt, np.ndarray):
                        rt = rt.astype(np.uint8).tobytes()
                    text = (rt.rstrip(b"\x00").decode("utf-8", "replace").strip()
                            if isinstance(rt, bytes) else str(rt).strip())
                    break
            # preprocessing to match the training dataloader
            if session_stats and session in session_stats:
                m = np.asarray(session_stats[session]["mean"])
                s = np.asarray(session_stats[session]["std"])
                neural = (neural - m) / (s + 1e-8)
            if sigma and gaussian_filter1d is not None:
                neural = gaussian_filter1d(neural, sigma=sigma, axis=0)
            by_session.setdefault(session, []).append({
                "neural": torch.tensor(neural, dtype=torch.float32),
                "ref": ref, "text": text, "session": session,
            })
    # chronological order + per-session cap
    out = OrderedDict()
    for sess in sorted(by_session.keys()):
        trials = by_session[sess]
        if max_per_session:
            trials = trials[:max_per_session]
        out[sess] = trials
    return out


@torch.no_grad()
def decode_phonemes(logits_fn, neural, session=None, blank=0):
    """Greedy CTC decode of one trial -> (phoneme-id list tensor, confidence)."""
    import torch.nn.functional as F
    from core.consolidator import ctc_greedy_decode
    x = neural if neural.dim() == 3 else neural.unsqueeze(0)
    logits = logits_fn(x, session_id=session, write_memory=False)
    lp = F.log_softmax(logits, dim=-1)
    hyp = ctc_greedy_decode(lp, blank=blank)[0]
    conf = float(lp.exp().max(dim=-1).values.mean().item())
    return hyp, conf
