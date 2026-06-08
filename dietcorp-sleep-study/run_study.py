"""
run_study.py
------------
Standalone driver for the sleep-consolidation study. Builds the trained CTC model, then
runs the condition x N-sweep x day grid and writes PER[condition][N][day] (+ wake/consolidate
latency). Decoupled from the brain2text-experiments registry/run.py.

Conditions:
  C0 no-adapt (N=0)            C1 self-label    C2 LM-refined
  C3 LM + memory + replay      C4 oracle labels

Usage:
  py -3 run_study.py --config configs/study.yaml
  py -3 run_study.py --checkpoint ".../best_model_per.pth" --data data/sessions \
      --conditions C0 C1 C2 --n_steps 0 1 2 4 --max_sessions 4 --max_trials 16
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from core.model import build_ctc_model, patch_embed_params
from core.episodic_memory import EpisodicBuffer
from core.consolidator import TTAConsolidator, TTAConfig
from core.replay import ReplayStore, ReplayItem
from core.drift_eval import load_real_sessions, decode_phonemes, per
from core.lm_refine import PhonemeNGramLM
from core.wer_decode import Lexicon, decode_words, compute_wer

CONDITIONS = {
    "C0": dict(mode="self", memory=False, replay=False, n_only=[0]),
    "C1": dict(mode="self", memory=False, replay=False),
    "C2": dict(mode="lm",   memory=False, replay=False),
    "C3": dict(mode="lm",   memory=True,  replay=True),
    "C4": dict(mode="oracle", memory=False, replay=False),
}


def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def run_condition(cond_id, base_state, model, days, target_param_names, lm, args,
                  lexicon=None, blank=0):
    spec = CONDITIONS[cond_id]
    n_list = spec.get("n_only", args.n_steps)
    out = {"by_n": {}, "wake_latency_ms": {}, "consolidate_ms": {}}

    for N in n_list:
        model.load_state_dict(base_state)                 # reset to baseline each N
        memory = None
        if spec["memory"]:
            memory = EpisodicBuffer(embed_dim=model.encoder.embed_dim,
                                    buffer_size=args.buffer_size).to(args.device)
        replay = ReplayStore(capacity=args.replay_capacity) if spec["replay"] else None
        tparams = patch_embed_params(model)
        cfg = TTAConfig(n_aug=args.n_aug, mask_frac=0.53, lr=args.lr,
                        confidence_threshold=0.0, lm_weight=args.lm_weight)
        cons = TTAConsolidator(model, tparams, memory=memory, lm=lm, config=cfg)

        # day-0 self-decode references (fallback when no oracle phonemes present)
        self_refs = {}
        first_day = next(iter(days.values()))
        for i, tr in enumerate(first_day):
            hyp, _ = decode_phonemes(cons.logits, tr["neural"].to(args.device), tr["session"], blank)
            self_refs[i] = hyp

        curve, wake, costs = [], [], []
        for day, trials in days.items():
            # evaluate BEFORE adapting on this day
            pers, confs, wers = [], [], []
            for i, tr in enumerate(trials):
                hyp, conf = decode_phonemes(cons.logits, tr["neural"].to(args.device),
                                            tr["session"], blank)
                ref = tr["ref"] if tr["ref"] is not None else self_refs.get(i, hyp)
                pers.append(per(hyp, ref)); confs.append(conf)
                if lexicon is not None and tr.get("text"):
                    words = decode_words(hyp, lexicon, lm_weight=args.wer_lm_weight)
                    wers.append(compute_wer(words, tr["text"]))
            curve.append({"day": day, "per": float(np.mean(pers)),
                          "wer": float(np.mean(wers)) if wers else None,
                          "confidence": float(np.mean(confs)), "n_trials": len(trials)})

            if N > 0:
                for i, tr in enumerate(trials):
                    neural = tr["neural"].to(args.device)
                    if spec["mode"] == "oracle":
                        if tr["ref"] is None:
                            continue                       # oracle needs labels
                        labels, conf = tr["ref"].to(args.device), 1.0
                    else:
                        labels, conf = cons.pseudo_label(neural, mode=spec["mode"],
                                                         session_id=tr["session"])
                    batch = [{"neural": neural, "labels": labels,
                              "session": tr["session"], "confidence": conf}]
                    if replay is not None and len(replay) > 0:
                        for it in replay.sample(args.replay_k):
                            batch.append({"neural": it.neural.to(args.device), "labels": it.ref,
                                          "session": it.session, "confidence": it.confidence})
                    m = cons.consolidate(batch, n_steps=N)
                    if not m["skipped"]:
                        wake.append(m["wake_latency_ms"]); costs.append(m["consolidate_ms"])
                        if replay is not None:
                            with torch.no_grad():
                                lat = model.encode(neural.unsqueeze(0) if neural.dim() == 2 else neural,
                                                   session_id=tr["session"]).mean(dim=(0, 1))
                            surprise = (m["loss_before"] or 0.0)
                            replay.add(ReplayItem(neural=neural.detach().cpu(), latent=lat.detach().cpu(),
                                                  confidence=conf, surprise=surprise,
                                                  session=tr["session"], ref=labels.detach().cpu()))
        out["by_n"][N] = curve
        out["wake_latency_ms"][N] = float(np.mean(wake)) if wake else None
        out["consolidate_ms"][N] = float(np.mean(costs)) if costs else None
        first, last = curve[0]["per"], curve[-1]["per"]
        wer_last = curve[-1].get("wer")
        wer_str = f" WER@last={wer_last:.4f}" if wer_last is not None else ""
        print(f"  [{cond_id}] N={N}: PER day0={first:.4f} last={last:.4f}{wer_str} "
              f"wake={out['wake_latency_ms'][N]} cons={out['consolidate_ms'][N]}")
    return out


def validate_mapping(model, days, lexicon, device, blank=0, max_trials=50):
    """Decode GROUND-TRUTH seq_class_ids -> words and report WER vs transcription.
    If the phoneme inventory (core/phonemes.py) is correct this is ~0; high WER means the
    inventory/lexicon does not match the data formatting (fix before trusting WER numbers)."""
    n, tot = 0, 0.0
    for trials in days.values():
        for tr in trials:
            if tr["ref"] is None or not tr.get("text"):
                continue
            words = decode_words(tr["ref"], lexicon)
            tot += compute_wer(words, tr["text"]); n += 1
            if n >= max_trials:
                break
        if n >= max_trials:
            break
    if n == 0:
        print("[validate] no oracle refs+text available to validate the mapping."); return
    print(f"[validate] oracle seq_class_ids -> words WER = {tot / n:.4f} over {n} trials "
          f"({'OK: inventory matches' if tot / n < 0.15 else 'WARNING: inventory likely WRONG'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint")
    ap.add_argument("--data")
    ap.add_argument("--session_stats", default=None)
    ap.add_argument("--lm", default=None)
    ap.add_argument("--lexicon", default=None, help="lexicon.json for word-level WER decoding")
    ap.add_argument("--word_decode", action="store_true", help="also compute word-level WER")
    ap.add_argument("--validate_mapping", action="store_true",
                    help="decode oracle seq_class_ids -> words and report WER (mapping self-check)")
    ap.add_argument("--wer_lm_weight", type=float, default=1.0)
    ap.add_argument("--conditions", nargs="*", default=["C0", "C1", "C2", "C3", "C4"])
    ap.add_argument("--n_steps", nargs="*", type=int, default=[0, 1, 2, 4, 8])
    ap.add_argument("--max_sessions", type=int, default=None)
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--n_aug", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lm_weight", type=float, default=0.5)
    ap.add_argument("--buffer_size", type=int, default=256)
    ap.add_argument("--replay_capacity", type=int, default=512)
    ap.add_argument("--replay_k", type=int, default=2)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.config:
        cfg = _load_yaml(args.config)
        for k, v in cfg.items():
            if getattr(args, k, None) in (None, [], False) or k in ("conditions", "n_steps"):
                setattr(args, k, v)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[study] device={args.device} conditions={args.conditions} N={args.n_steps}")

    session_stats = None
    if args.session_stats and os.path.exists(args.session_stats):
        with open(args.session_stats) as f:
            session_stats = json.load(f)

    model = build_ctc_model(args.checkpoint, device=args.device)
    base_state = copy.deepcopy(model.state_dict())

    days = load_real_sessions(args.data, split="val", session_stats=session_stats,
                              max_per_session=args.max_trials)
    if args.max_sessions:
        from collections import OrderedDict
        days = OrderedDict(list(days.items())[:args.max_sessions])
    print(f"[study] {len(days)} sessions (days): {list(days.keys())}")

    lm = PhonemeNGramLM.load(args.lm) if (args.lm and os.path.exists(args.lm)) else None
    if any(c in ("C2", "C3") for c in args.conditions) and lm is None:
        print("[study] WARNING: C2/C3 requested but no LM loaded — falling back to self-label.")

    lexicon = None
    if (args.word_decode or args.validate_mapping) and args.lexicon and os.path.exists(args.lexicon):
        lexicon = Lexicon.load(args.lexicon)
        print(f"[study] loaded lexicon ({len(lexicon.prons)} words) for WER decoding")
    elif args.word_decode:
        print("[study] WARNING: --word_decode set but no lexicon found — WER skipped.")

    if args.validate_mapping and lexicon is not None:
        validate_mapping(model, days, lexicon, args.device)

    results = {}
    for cond in args.conditions:
        print(f"[study] === condition {cond} ===")
        results[cond] = run_condition(cond, base_state, model, days,
                                      patch_embed_params(model), lm, args,
                                      lexicon=lexicon if args.word_decode else None)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "study_results.json")
    merged = {}
    if os.path.exists(out_path):                  # merge so multi-invocation runs accumulate
        try:
            with open(out_path) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(results)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[study] wrote {out_path} (conditions: {sorted(merged.keys())})")


if __name__ == "__main__":
    main()
