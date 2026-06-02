# Autoresearch Feasibility & Validation Audit

**Prepared:** 2026-06-03
**Target framework:** `brain2text-experiments` (`run.py` + `registry.yaml` + `stack.py` + `compose.py`)
**Scope:** Tracks A, B, D, E (the locally-runnable autoresearch sweep)

This is the feasibility map for the autoresearch loop defined in `autoresearch/program.md`.
It records which experiments actually run on the RTX 4050 / Windows box today, which are gated,
and every bottleneck that would break an unattended sweep. The executable counterpart is
`autoresearch/preflight.py` (§6) — it turns this table into a re-runnable check.

**Headline:** unlike a from-scratch harness, most infrastructure already exists. `registry.yaml`
defines 25+ experiments with `local_ok`, `depends_on`, `expected_wer_band`, `expected_runtime_min`,
and maturity `state` flags. `run.py` enforces pytest → toy → full, runs a step-50 `SmokeAssert`,
and logs `WER@10`/slope/`best_wer` to `leaderboard.sqlite`. Tracks D and E are **already wired as
specs** (no new CLI flags or projector registry needed). The toy dataset already exists. The
autoresearch layer is therefore a thin orchestration loop over `run.py`, not new training code.

---

## 1. Entry point and the existing gates

```
python run.py --expt <ID> --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
                          [--override key=value ...]   # spec mutation surface
```

Three validation layers already ship — the loop reuses them, never reimplements them:

| Gate | Mechanism | Catches |
|---|---|---|
| **Shape gate** (CPU, ~10 s) | `pytest tests/test_stage_shapes.py` | encoder/projector/decoder shape mismatches before any GPU use |
| **Smoke assert** (step 50) | `run.py` `SmokeAssert` | CE≥13, CTC≥5, NaNs, throughput<50 tok/s → fast fail |
| **Validity band** (post-run) | `expected_wer_band` in `registry.yaml` | results outside the plausible range flagged SUSPECT |

Promotion to `--profile full` is hard-blocked unless a toy pass exists in `leaderboard.sqlite`
within 7 days. The autoresearch loop never calls `full` itself — that stays a human decision.

---

## 2. Runnability by experiment (verified against `registry.yaml`)

`local_ok` and `state` are read straight from the registry. "Runs today" = `local_ok: true`,
no `skeleton`/`partial` state, dependencies satisfiable locally.

| Exp | Track | local_ok | state | Runs today? | Note |
|---|---|---|---|---|---|
| **A1** CKA | A | ✅ | — | ⚠️ partial | 1.5B/3B rows local; 7B rows cloud (B1 below) |
| **A2** Perplexity | A | ✅ | — | ⚠️ partial | same VRAM caveat for 7B |
| **A3** Phoneme probe | A | ✅ | — | ✅ | tiny probe; frozen LLM hidden states |
| **A4** Audio-vs-vision E2E | A | ❌ | — | ☁️ cloud | 7B models; `depends_on A1/A2/A3` |
| **B0_baseline** | B | ✅ | — | ✅ | the from-scratch control — run first |
| **B1** Conformer | B | ✅ | — | ✅ | `depends_on B0` |
| **B2** HRM | B | ✅ | — | ✅ (perf-gated) | DEQ cost — see C4 |
| **B3** Mamba (GRU) | B | ✅ | — | ✅ | GRU fallback, pure PyTorch |
| **B3_mamba** true SSM | B | ✅* | — | ☁️ cloud/Linux | `mamba-ssm` won't build on Windows — see C1 |
| **B4** MoE | B | ✅ | — | ✅ | aux loss set in spec — see C5 |
| **B5** ZenBrain | B | ✅ | — | ✅ | `use_memory` training path |
| **C3** Whisper-Qwen | C | ✅ | — | ✅ (Track C, optional) | ~4.1 GB; only local C-track |
| **C1/C2** | C | ❌ | — | ☁️ cloud | 7B / 5.6B |
| **D1b D1d D2a D2d D3b D3c D4** | D | ✅ | — | ✅ | all wired as specs + `compose.py` |
| **E1a E1b E2b** | E | ✅ | — | ✅ | projector stages in `stack.py` |
| **E3** patch×query grid | E | ✅ | — | ✅ (~2 h) | `depends_on E2b`; 6-cell grid |
| **F1/F2/F3** JEPA | F | ✅ | **skeleton** | ⛔ skip | patchifier stubbed; out of A/B/D/E scope |
| **G1–G3, H1–H2** | G/H | ✅ | **partial** | separate thesis | driven by `run_thesis.ps1`, not this loop |

\* `B3_mamba` is `local_ok: true` in the registry but blocked in practice by C1 on Windows.

---

## 3. Bottleneck register

Severity: 🔴 blocks · 🟠 workaround · 🟡 watch.

| ID | Where | Bottleneck | Sev | Mitigation |
|---|---|---|---|---|
| **C1** | B3_mamba | `mamba-ssm` + `causal-conv1d` need a CUDA build toolchain absent on native Windows. | 🔴 | Registry already splits `B3` (GRU, local) from `B3_mamba` (cloud). Sweep runs `B3`; `B3_mamba` deferred to A100/Linux. |
| **C2** | A1/A2 7B rows, A4 | 7B in fp16 ≈ 14 GB; A1 forbids quantization (it measures embedding geometry). 6 GB can't hold them. | 🔴 (7B only) | Run 1.5B/3B rows local; 7B CKA + `A4` → cloud. Document that quantizing to fit corrupts what A1 measures. |
| **C3** | All LLM-decoder runs | `bitsandbytes` 4-bit NF4 on Windows historically fragile; local path only ever exercised on A100. | 🔴 if import fails | Preflight L0 does one 4-bit forward. If it fails: run encoder-only + `--adapt` experiments, defer LLM-decoder runs to cloud, or pin a known-good Windows wheel. |
| **C4** | B2 HRM | DEQ fixed-point solver (≤10 iters/patch) inflates runtime well past the 30-min registry estimate. | 🟠 | Perf cap: if epoch-1 time > 3× `B0`, switch to wall-clock budget, rank on WER-at-time. |
| **C5** | B4 MoE | Expert collapse without the load-balance aux loss. | 🟠 | `B4` spec sets `aux_loss_weight`; preflight asserts it > 0 and that the encoder exposes `last_aux_loss`. |
| **C6** | Track A 7-day full gate | `run.py` blocks `full` without a recent toy pass — but Track A is analysis (no toy WER). | 🟡 | A-track produces CKA/PPL/probe artefacts, not leaderboard WER rows; the loop treats A as terminal-local, never promotes it to `full`. |
| **E1** | Windows console | cp1252 console raises `UnicodeEncodeError` on non-ASCII log chars → kills an unattended run. | 🟡 | `$env:PYTHONIOENCODING="utf-8"` per subprocess (already done in `run_thesis.ps1`); logs to UTF-8 files. |
| **E2** | Windows dataloader | `num_workers>0` spawn-pickling deadlock risk. | 🟡 | Force `num_workers=0`. |
| **E3** | Checkpoint dependency | G2/G3 and any SSL-warm-start path need `best_model_per.pth`; B1–B5 deliberately run from scratch. | 🟡 | A/B/D/E sweep needs no checkpoint (B0 is the from-scratch control). Only flag if a spec references a missing `pretrained_ckpt`. |

---

## 4. What is *already solved* (and was on the earlier roadmap)

The first audit (written against the sibling `brain2text-modeltraining-multiarchitectures` repo)
flagged these as work — they are already done **here**:

- **Toy dataset** → `tools/make_toy_dataset.py` + `data/toy_train.hdf5` (2 GB) / `toy_val.hdf5` (0.5 GB).
- **Track D wiring** → `D1b/D1d/D2a/D2d/D3b/D3c/D4` are registry specs composed by `compose.py`
  (no `--ctc_weight`/`--label_smoothing` CLI flags needed; loss config lives in the spec).
- **Track E wiring** → `E1a/E1b/E2b/E3` are registry specs with projector stages in `stack.py`
  (no separate projector registry needed).
- **Smoke/contract gates** → `SmokeAssert` + `tests/test_stage_shapes.py`.
- **Metric logging** → `leaderboard.sqlite` with `WER@10`, `slope`, `best_wer`, `spec_hash`, `code_hash`.

So the autoresearch build reduces to: a **preflight reader** (§6) + a **sweep loop** that wraps
`run.py` and applies the §5 policy from `program.md` + a **presentation digest**.

---

## 5. Validation protocol for the sweep

```
once     PREFLIGHT   env probe (torch/CUDA/bitsandbytes 4-bit) + registry parse + shape gate
                     → writes autoresearch/runnable.json  (runnable | deferred-cloud | skeleton)
per-exp  L1 SMOKE    run.py step-50 SmokeAssert (built in) — fast fail, log reason
per-exp  L2 TOY      ~20-min toy run → leaderboard row (WER@10, slope, best_wer)
post     VALIDATE    best_wer ∈ expected_wer_band ? else flag SUSPECT
post     RANK        slope vs track baseline → PROMOTE / COMBINE / HOLD / DISCARD (program.md §5)
```

Track baselines: B vs `B0_baseline`; D and E vs the BIT + default-loss + MLP row.

---

## 6. What `preflight.py` adds (the only new validation code)

A thin reader — most checking already exists, so this stays small:

1. **Env probe (once):** import torch/transformers/peft; CUDA visible; **one 4-bit `bnb.nn.Linear`
   forward** (catches C3); record VRAM headroom.
2. **Registry parse:** load `registry.yaml`; for each A/B/D/E experiment classify
   `runnable_local` / `deferred_cloud` / `skeleton` from `local_ok` + `state` + `depends_on` +
   the bottleneck rules (C1 forces `B3_mamba`→cloud; C2 forces 7B rows→cloud).
3. **Shape gate:** invoke `pytest tests/test_stage_shapes.py` once; record pass/fail per stage.
4. **Spec sanity:** for each runnable exp, assert referenced `pretrained_ckpt` exists (if any) and
   `B4`'s `aux_loss_weight > 0`.
5. **Write `autoresearch/runnable.json`** — the manifest the sweep loop consumes:
   ```json
   {"expt": "B4", "track": "B", "runnable_local": true, "blockers": [],
    "depends_on": ["B0_baseline"], "expected_wer_band": [0.18, 0.50],
    "notes": "aux_loss_weight verified > 0"}
   ```

When bitsandbytes is fixed or a new spec lands, re-running preflight updates the map — the
feasibility table stays live rather than going stale.

---

## 7. Verdict by track

| Track | Local verdict | Caveats |
|---|---|---|
| **A** | ⚠️ Partial | A3 clean; A1/A2 local for 1.5B/3B, cloud for 7B (C2); A4 cloud. Analysis-only, never promoted to `full`. |
| **B** | ✅ Today | B0 control + B1/B2/B4/B5 clean; B3 GRU-only local (C1); B2 perf-gated (C4). |
| **D** | ✅ Today | All seven specs wired; no VRAM concern. |
| **E** | ✅ Today | E1a/E1b/E2b clean; E3 is a ~2 h grid (run last, depends on E2b). |

**Milestone-1 sweep (all runnable today):** `B0_baseline → B1 B2 B3 B4 B5 → D1b D1d D2a D2d D3b
D3c D4 → E1a E1b E2b → E3 → combination phase`. Track A analysis runs alongside (no training budget).

---

*Companion: `autoresearch/program.md` (the steering doc). Next build: `autoresearch/preflight.py`
then the sweep loop wrapping `run.py`.*
