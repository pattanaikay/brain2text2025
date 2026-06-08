# Plan v3 — Standalone sleep-consolidation study on A100 (real DietCorp data)

## Context

The thesis needs a clean, defensible proof on **real** neural data, and the pieces that make
it work are currently missing or scattered:
1. The episodic buffer is **not** in the adaptation loop (only in H2 training).
2. N>1 self-labeled TTA **collapses** (proven: PER 0.64→4.2 for N=1→8) — pseudo-label noise
   compounds; DietCorp avoids this with a 3-gram LM, which we don't yet have.
3. All prior runs used **synthetic** drift; the real 17 GB `preprocessed_data.h5` has dated
   sessions = true chronological drift, unused so far.

**Decisions locked:**
- **Full stack** proof: LM-refined labels + episodic-memory anchor + priority replay + oracle ceiling.
- **Gradient-descent N-step** consolidation (DietCorp's rule generalized to N) — **not** the sleep
  paper's learned-local-rule arm.
- **A100-only execution.** Local machine = code dev + CPU unit tests only.
- **Fully self-contained, standalone folder**, decoupled from the existing registry/`run.py` sweeps:
  it **copies** the needed modules (no import dependency on `brain2text-experiments`).

**What we take from the sleep paper (be explicit in the defense):**
- USE: offline N-pass consolidation at the eviction boundary (= between sentences); strict
  wake/sleep separation (wake = one forward pass, all extra compute offline); **N as the scaling
  variable**, with the prediction that gains grow with N on *sequential* problems (drift is sequential).
- DO NOT use: their learned local rule trained end-to-end through the N-loop. Our rule is fixed
  gradient descent on a CTC pseudo-label.

**Grounding facts:**
- `brain2text-modeltraining/data/preprocessed_data.h5`: 10,948 trials, keys
  `t15.YYYY.MM.DD_..._trial_NNNN` (`neural` + `transcription`); sessions dated → real drift.
- Formatted trials carry `seq_class_ids` (42-class phoneme IDs); g2p/phoneme + WER logic in
  `src/preprocessing/dataloader.py`, `src/utils/metrics.py` → copy for oracle labels + WER.
- Checkpoint `best_model_per.pth`: `encoder.*` (46 sessions) + `head.weight/bias` (42 phonemes).

---

## The standalone package: `dietcorp-sleep-study/`  (sibling to brain2text-experiments)

```
dietcorp-sleep-study/
  core/
    consolidator.py     # COPY adapt/dietcorp_tta.py + wire memory anchor + replay + LM hook
    episodic_memory.py  # COPY stages/memory/episodic_buffer.py + stages/loss/episodic_consistency.py
    lm_refine.py        # NEW  n-gram pseudo-label refiner (KenLM on A100; py-ngram fallback)
    replay.py           # NEW  Simulation-Selection: priority = |CTC-surprise| + (1-conf) + novelty
    drift_eval.py       # COPY tools/drift_eval.py + load_real_sessions(h5, split="val")
    phonemes.py         # COPY the 42-class g2p + phoneme<->word mapping from src/...
    model.py            # encoder(BIT) + ctc_head loader (COPY the bit.py + head-load logic)
  configs/
    study.yaml          # conditions C0..C4, N-sweep list, paths, drift source=real
  run_study.py          # STANDALONE driver: builds model, loops conditions x N x days -> WER grid
  a100/
    env_setup.sh        # venv, torch cu121, kenlm, deps  (pinned requirements.txt)
    prepare_data.py     # verify preprocessed_data.h5; build phoneme + word n-gram LMs once
    build_lm.py         # train n-gram LMs from transcriptions (g2p)
    run_matrix.sh       # run C0..C4 in sequence, gated, auto-collect
    collect_and_plot.py # WER-vs-day figure (one curve per N) + results table per condition
    SYNC.md             # create fresh A100 (jarvislabs skill / jl CLI), upload data+ckpt+code
  tests/                # CPU unit tests (mirror existing + new lm_refine/replay)
  RESULTS.md            # filled after the run
```

---

## Components

### 1. `core/consolidator.py`  (copy + extend `adapt/dietcorp_tta.py`)
- Keep: `augment` (time-mask 64×), `consolidate(N steps on patch-embed only)`, wake/consolidate timing.
- Extend `pseudo_label(neural, mode="self"|"lm"|"oracle", refiner=None, true_labels=None)`.
- Add optional `memory` (EpisodicBuffer) + `episodic_weight`: wake forward routes tokens through
  memory (cross-attn read) before `ctc_head`; consolidation loss = CTC + `episodic_weight`·MSE(query, retrieved.detach()).

### 2. `core/lm_refine.py`  (NEW — the critical fix)
- Phoneme n-gram LM; CTC-beam + shallow-fusion rescoring → denoised pseudo-label. KenLM on A100,
  pure-Python n-gram fallback for local dev. Trained by `a100/build_lm.py` from g2p'd transcriptions.

### 3. `core/replay.py`  (NEW — ZenBrain Simulation-Selection)
- Host-side capped store of past trials; scheduler scores `|CTC-surprise| + (1-confidence) + novelty`,
  samples top-K for consolidation so each step trains on the most informative history.

### 4. `core/drift_eval.py`  (copy + add real loader)
- `load_real_sessions(h5, split="val")` → `OrderedDict[date → [(neural, phoneme_ref)]]`, chronological.
- `run_grid(model, days, conditions, n_steps_list)` → `WER[condition][N][day]` + wake/consolidate ms.

### 5. `run_study.py`  (standalone driver — replaces registry/run.py for this study)
Conditions:
| ID | Condition | Proves |
|---|---|---|
| C0 | no-adapt (N=0) | the drift baseline (22.7%→66.5% shape) |
| C1 | self-label, N∈{1,2,4,8} | does collapse persist on real data? |
| C2 | **LM-refined**, N∈{1,2,4,8} | **H_main**: deeper N helps with good labels |
| C3 | **LM + memory + replay**, N∈{1,2,4,8} | the ZenBrain contribution |
| C4 | **oracle**, N∈{1,2,4,8} | the achievable ceiling |
Output: `WER[day, N, condition]` grid (PER too, as the cheap proxy), JSON + the figure.

---

## What proves the thesis (decision gates)
- **Headline figure:** WER-vs-held-out-day, one curve per N, per condition.
- **H_main confirmed** iff C2 (LM) N>1 < C1/DietCorp N=1 at later days, **wake latency flat across N**.
- **Memory confirmed** iff C3 < C2 (esp. session starts / hardest-drift days).
- **C4 oracle** sets the ceiling and attributes residual gap to label quality.

## Verification
- **Local (CPU):** `py -3 -m pytest dietcorp-sleep-study/tests -q` — consolidator, lm_refine,
  replay, real-session loader all green.
- **Local (GPU smoke):** `run_study.py --condition C2 --sessions 4 --n_steps 0 1 2` on a 4-session
  slice → LM-refined N-sweep no longer collapses (N≥2 ≤ N=1).
- **A100 (headline):** `a100/run_matrix.sh` → full grid + WER-vs-day figure; compare C2/C3 vs
  C1 (N=1 DietCorp) and the published 12.17%.

## Risks
- **KenLM on Windows** → py-ngram locally; KenLM only on Linux A100.
- **g2p / 42-class mapping must match the trained head** → copy exactly from `dataloader.py`/checkpoint.
- **Replay store memory** → cap raw-trial store; latents for read, raw only for replay.
- **17 GB upload + runtime** → first A100 pass on val splits / a session subset; gate C3/C4 on C2.