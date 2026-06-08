# program.md — Autoresearch Steering Document

> This is the human-editable brain of the autoresearch loop, modelled on Karpathy's
> `autoresearch/program.md`. The harness (and any agent driving it) reads **this file**
> to decide what to run next, how to judge a result, and when to stop. **You steer the
> research by editing this file — not by editing `run.py`, `stack.py`, or the specs.**
>
> Everything below the line "AGENT: READ FROM HERE" is the operative contract.

---

## 0. The research thesis (why this loop exists)

I have reproduced the base pipeline from Zhang et al. (2025):

```
neural activity → BIT encoder → MLP projector → Qwen2.5-1.5B (QLoRA) → sentence
losses: CE  +  0.3·CTC  +  1.0·InfoNCE-contrastive
```

Baseline: **WER 0.3673**. Target: **WER < 0.10**.

The compact mental model (from `zhangpaperdiscussion.md`) is the spine of the whole search:

```
Encoder:   learns speech-relevant neural time patterns.
Projector: translates neural tokens into LLM-compatible embeddings.
LLM:       turns those embeddings into English text.
Losses:    tell each part what "useful" means.
QLoRA:     adapts the LLM cheaply to the neural input modality.
```

**The autoresearch goal is not to find one clever trick. It is to map which of the four
levers — encoder, projector, losses, (and decoder, cloud-only) — actually move WER, and
then compose the winners into an optimal end-to-end model.** Each lever is one track. The
loop runs them cheaply on toy data, ranks them by learning signal, and promotes only what
earns it.

---

## 1. How this maps onto Karpathy's autoresearch

| Karpathy autoresearch | This repo |
|---|---|
| `train.py` — the one file an agent mutates | **`run.py --expt <ID>`** + **`--override key=value`** — the mutation surface |
| `prepare.py` — fixed scaffold | `registry.yaml` + `specs/*.yaml` + `stack.py` + `compose.py` + the toy HDF5 |
| `program.md` — human steering | **this file** |
| metric: `val_bpb` (lower better) | **WER@10 + slope + best_wer** from `leaderboard.sqlite` (lower better) |
| fixed 5-min budget | `--profile toy` (≈20 min/run); perf-gated archs get a wall-clock cap (§6) |
| keep / discard | the promote/discard policy in §5 |
| ~100 runs overnight | one curated sweep of Tracks A/B/D/E + a combination phase |

The key difference: the search space here is **curated** by `registry.yaml`, not free-form
code. That is deliberate — every experiment is shape-gated, smoke-asserted, and WER-banded
before it runs. The agent's freedom lives in (a) **which** registry experiment to run next
and (b) **`--override`** mutations that compose or tune winners.

---

## 2. The mutation surface

The harness/agent may only change the run in these three ways. Nothing else.

1. **Pick a registry experiment**: `python run.py --expt <ID> --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5`
2. **Override spec scalars**: `--override encoder.patch_size=5 projector.n_queries=32 loss.ctc_anneal.start_weight=0.15`
   — used to *tune* a winner or *compose* two winners (the combination phase, §5.3).
3. **Adapt-mode** (Track G/drift only, out of A/B/D/E scope): `--adapt --n_steps 0 1 2 4`.

`run.py` enforces the **pytest → toy → full** progression, but this loop is **toy-only**: it
**never** runs `--profile full` or any 150-epoch run. Every experiment is a short, diagnostic
toy run on the A100. The aim is to identify the best building block per stage; training the
winning combination to convergence is a **separate** downstream goal, not part of this loop.

---

<!-- ═══════════════════════════════════════════════════════════════════════ -->
# AGENT: READ FROM HERE
<!-- ═══════════════════════════════════════════════════════════════════════ -->

## 3. What to run, in what order

Everything runs **on the A100** (40 GB) — nothing local. Respect `depends_on`. Scope = Tracks
**A, B, C, D, E, F** (Track C — C1/C2/C3 — fits the A100; A4 and B3_mamba likewise). **Track F
(JEPA) is ACTIVE:** `stages/encoder/jepa.py` now has real per-modality backbones — a wav2vec2-style
1D temporal conv stack (audio), a DINOv2-style 2D patch conv (video), and a native patch-embed
(neural control) — trained via JEPA under the controlled A/B (F1/F2/F3 byte-identical except
`modality:`). F is a *pretraining* track: it yields a backbone + downstream eval, not a single toy
WER (phase 6). The loop stays toy-only throughout. Tracks G/H are the separate DietCorp/ZenBrain
thesis (driven by `run_thesis.ps1`).

**Phase order (cheap → expensive):**

1. **Track A — analysis (~1.5 h).** `A1` CKA, `A2` perplexity, `A3` phoneme probe, plus `A4`
   (E2E decoder-modality comparison). `A1`/`A2` are tools (`tools/cka_analysis.py`,
   `tools/perplexity_test.py`); `A3` trains a tiny probe. All rows fit the A100 (incl. 7B).
2. **Track B — encoder sweep.** Run `B0_baseline` **first** (the fair from-scratch control every
   B1–B5 is judged against). Then `B1 B2 B3 B3_mamba B4 B5` in any order.
3. **Track C — decoder sweep.** `C1` Qwen2-Audio-7B, `C2` Phi-4-MM, `C3` Whisper-Qwen. BIT
   encoder fixed; only the LLM decoder changes.
4. **Track D — loss sweep.** `D1b D1d D2a D2d D3b D3c D4`. Same architecture (BIT), so
   differences isolate the loss term. Independent — any order.
5. **Track E — projector sweep.** `E1a E1b E2b`, then `E3` (depends on `E2b`; 6-cell
   patch×query grid, run last).
6. **Track F — JEPA pretraining (controlled A/B/C).** Integrity gate FIRST
   (`tests/test_jepa_smoke.py` + `tools/diff_specs.py` — specs differ only in `modality:`), then
   pretrain `F1` (audio) → `F2` (video) → `F3` (neural); evaluate each backbone downstream. Ranked
   on pretraining health (no collapse) + downstream decoding, not a single toy WER.
7. **Combination phase (§5.3).** Compose the Track-B/C/D/E winners via `--override` and re-run
   toy. This is where the "optimal E2E model" is assembled.

After each run, read the new `leaderboard.sqlite` row and apply §5.

---

## 4. The metric (how to read a result)

From `leaderboard.sqlite`, per (expt, toy) run:

- **`best_wer`** — lowest WER across the toy run.
- **`wer_epoch`** — where best occurred.
- **`slope`** — WER improvement rate (the primary ranking signal on toy, because absolute
  toy WER is noisy and B1–B5 start from scratch with no SSL checkpoint).

Rank within a track by **slope first, best_wer as tiebreak**. Always compare a Track-B arch
against **`B0_baseline`** (from-scratch BIT), never against the production 0.3673 number.

**Validity gate:** if `best_wer` falls outside the experiment's `expected_wer_band` in
`registry.yaml`, flag the row as SUSPECT (likely a broken run, not a real result) and do not
promote it — investigate instead.

---

## 5. Promote / discard policy

Computed on **toy slope improvement vs. the track baseline** (Track B vs `B0_baseline`;
Tracks D/E vs the BIT+default-loss+MLP baseline row):

```
slope improvement ≥ 5% relative   → STRONG:    flag as a best-building-block for the separate
                                                long-term goal (NO full run executed here)
slope improvement   3–5% relative  → PROMISING: carry into the combination phase, re-run toy
slope improvement   1–3% relative  → WEAK:      keep only if it helps a combination
slope improvement  < 1% relative   → INERT:     log the negative result, move on
```

**No full runs.** "STRONG" never triggers a 150-epoch run in this loop — it only marks the
component as a winner to hand off to the separate long-term-implementation effort.

Additional rules:
- **Negative results are kept**, not deleted — a clean "D2a (no contrastive) ties D2c" is a
  publishable finding (the contrastive loss may be inert or harmful).
- **Val-loss divergence**: if val loss rises faster than baseline while WER improves, it is a
  calibration artefact — acceptable, promote anyway. If val loss diverges AND WER stalls →
  overfitting, add dropout before promoting.

### 5.3 Combination phase (assembling the optimal E2E model)
Take the **best encoder** (Track B), **best projector** (Track E), **best loss config**
(Track D). Compose them in one toy run via `--override` on the winning encoder's spec, e.g.:

```
python run.py --expt B1 --profile toy \
  --override projector.variant=qformer projector.n_queries=32 \
             loss.ctc_anneal.anneal_epochs=75 loss.contrastive.weight=0.0 \
  --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
# override keys must match the specs: projector.variant (not "kind");
# loss entries keyed by variant — loss.ctc_anneal.* / loss.contrastive.*
```

If the combination's slope beats each ingredient alone → that is the candidate for the
headline cloud run. If it underperforms the best single lever → the levers interact
negatively; report it and promote the best single lever instead.

---

## 6. Budget & safety (two-tier)

- Every experiment is already shape-gated (`tests/test_stage_shapes.py`) and smoke-asserted
  at training step 50 (CE<13, CTC<5, no NaNs, throughput≥50 tok/s) by `run.py`. The
  autoresearch loop relies on these — it does **not** reimplement them.
- **Tier 1 (smoke):** the step-50 SmokeAssert is the fast-fail. If a run dies there, record
  the reason and move on — do not retry blindly.
- **Tier 2 (toy):** full ~20-min toy run → leaderboard row.
- **Perf cap:** `B2` (HRM, DEQ fixed-point) can run far slower than nominal. If its first
  epoch exceeds ~3× the `B0` per-epoch time, cap it by wall-clock and rank on WER-at-time.

**Windows execution (non-negotiable):**
- Launch with **`py -3`** (not `python` — that's the Store stub).
- Set **`$env:PYTHONIOENCODING="utf-8"`** in every subprocess (cp1252 console crashes on
  non-ASCII log chars — this *will* kill an unattended overnight run otherwise).
- Force `num_workers=0` for dataloaders on Windows (spawn-pickling deadlock risk).
- One `output_dir` per run; never share.

---

## 7. Known bottlenecks (carry these into every decision)

| ID | Where | What | Action |
|---|---|---|---|
| H1 | **B3_mamba**, Track C | `mamba-ssm`/`causal-conv1d` don't build on Windows; 7B decoders exceed 6 GB. | **Resolved on A100:** `pip install mamba-ssm causal-conv1d` works on Linux; 7B fits 40 GB. Preflight confirms both. |
| H2 | **A1/A4 7B rows** | A1 forbids quantization (it measures embedding geometry); 7B fp16 ≈ 14 GB. | **Resolved on A100** (40 GB holds 7B in fp16). |
| H3 | **bitsandbytes 4-bit** | The E2E LLM path depends on it. | Works out-of-box on the A100 PyTorch template; preflight does one 4-bit forward to confirm. |
| H4 | **B2 HRM cost** | DEQ iterations inflate runtime. | Perf cap (§6). |
| H5 | **B4 MoE** | Expert collapse without the load-balance aux loss. | The `B4` spec already sets `aux_loss_weight`; preflight asserts it is > 0. |

Full reasoning: `autoresearch/FEASIBILITY_AUDIT.md`.

---

## 8. Definition of done for one autoresearch cycle

1. Every `local_ok` A/B/C/D/E experiment has a `leaderboard.sqlite` toy row (or a logged
   skip-reason); each Track-F experiment has a JEPA pretraining record + downstream eval
   (F yields a backbone, not a toy WER).
2. Each track has a ranked winner with a slope-vs-baseline number.
3. The combination phase has produced one composed candidate.
4. A short results digest is written for the presentation layer (tables + a tufte/seaborn
   WER-by-track chart + a slope scatter) and folded into `research_summary.html`.
5. The BEST-BUILDING-BLOCKS list (the single best encoder / decoder / loss / projector by slope)
   is handed off for the separate long-term-implementation goal. No full run is executed here.

---

## 9. How to steer (edit these, re-run the loop)

- **Change scope:** edit the track list in §3.
- **Change aggressiveness:** edit the thresholds in §5.
- **Add a hypothesis:** add an entry to `registry.yaml` (+ a `specs/*.yaml`) and list its ID
  in §3. The loop picks it up automatically.
- **Force a combination:** write the exact `--override` string into §5.3.
- **Pause a track:** remove its IDs from §3 or set the experiments' `local_ok: false`.

*The loop is deterministic and re-runnable. Re-running skips experiments that already have a
fresh leaderboard row unless `--override` changes the config hash.*
