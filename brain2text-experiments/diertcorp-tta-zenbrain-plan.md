# Plan — Sleep-style memory consolidation in DietCorp TTA for drift-robust real-time neural decoding

## Context

**The thesis question (your words):** *does memory consolidation of the back-propagation
process used in the DietCorp process help us run real-time neural decoding better?*

Restated as a falsifiable hypothesis the codebase can answer:

> **H_main —** Increasing the *depth* of the per-trial / per-session consolidation step
> (N recurrent "sleep" passes instead of DietCorp's single AdamW step) reduces WER under
> day-to-day electrode drift, **while holding wake-time inference at a single forward pass.**

Why this is the right framing — the three sources line up on one axis:

| Source | What it contributes | Role of N (consolidation depth) |
|---|---|---|
| **DietCorp** (arXiv:2507.02800) | Per-trial TTA: 64 time-masked augs → n-gram/CTC pseudo-label → **one** AdamW step on the patch-embed module. Fights drift (22.7%→66.5% over 8 days becomes flat). | N = 1 (hand-designed, single step) |
| **Do Language Models Need Sleep** (arXiv:2605.26099) | At the eviction boundary, run **N offline recurrent passes** to refine fast-weights before clearing the cache; backprop through the whole consolidate→predict graph; **wake-time stays one pass**. Gains grow with N on the *deepest/most sequential* problems. | N > 1, and the rule is *learned* |
| **ZenBrain** (arXiv:2604.23878) | A memory hierarchy + **Simulation-Selection sleep loop**: offline replay whose priority = `\|TD\| + reward + novelty`; +37% stability, −47% storage. Tells us *what* to consolidate and *when*. | The scheduler around the consolidation step |

DietCorp's TTA **is** an N=1 special case of the sleep paper's consolidation. Neural drift
is a *sequential* problem (today's signal depends on yesterday's electrode state) — exactly
the regime where the sleep paper predicts deeper consolidation helps. The clinical objectives
in `research_summary.html` Project II (#2 streaming real-time, #4 drift robustness) are
precisely "improve drift WER **without** paying wake-time latency" — the sleep paper's core
promise. So the experiment is well-posed and the existing scaffolding was built for it.

**Assumptions (my recommended defaults for the four forks — flip any before I start):**
1. **DietCorp path:** native port on the existing toy HDF5 now + build the TTA loop natively;
   vendor upstream as a *read-only reference* (don't block on running their Linux/full-data loop).
2. **First consolidation prototype:** multi-step TTA (sweep N) — cheapest, most direct test of H_main.
3. **Eval scope:** local toy proof-of-mechanism first; design (not yet run) the cloud headline drift curve.
4. **ZenBrain memory policy:** session-keyed + confidence-gated; flip H1 skeleton→partial, make the episodic loss live.

---

## What exists vs. what's missing (grounded in the repo)

**Framework (works today):** `run.py` (train loop), `stack.py` (`Stack`: encoder→*memory?*→projector→decoder,
shape-checked at build), `compose.py` (`ComposedLoss` sums named losses). Confirmed runnable:
B0 baseline built Qwen2.5-1.5B and ran 50 steps on `data/toy_train.hdf5` before tripping the
smoke gate (CE 10.75 > 10.0). Env: `py -3` → Python 3.11, torch 2.5.1+cu121, CUDA on (RTX 4050).

**Track G (DietCorp) — present:**
- `docks/dietcorp_dock.py` — lazy/guarded upstream adapter, `upstream_hash()`, `run_smoke_iters()`.
- `stages/projector/dietcorp_recal.py` — **real** corrupted-input reg + per-day affine recal. ✅
- `tools/dietcorp_paper_oracle.yaml` — frozen-metric placeholder (nulls).
- `specs/G1_dietcorp_smoke.yaml`, `docks/PINS.txt` (`dietcorp_upstream … SHA_PENDING`).
- **GAP:** the **DietCorp TTA loop itself is NOT implemented** (no augmentations, no pseudo-label,
  no per-trial gradient step). This is the "backpropagation process" the thesis is about.

**Track H (ZenBrain) — present:**
- `stages/memory/episodic_buffer.py` — ring buffer + cross-attn read head, **guarded stub**
  (`guard_stub_forward`); `EpisodicWritePolicy` is a typed no-op enumerating FIFO/confidence/session.
- `stages/loss/episodic_consistency.py` — **backprop-safe zero** (`* 0.0`); delete to go live.
- `tests/test_zenbrain_stub_tripwire.py` — fails loudly if forward() is implemented without flipping state.
- `specs/H1_*.yaml` + `.health.json` (state: skeleton).
- **GAP:** no write policy, no `memory_query`/`memory_retrieved` wired into outputs, loss inert.

**Cross-cutting GAPs:** (a) no N-step "sleep" consolidation; (b) no drift-eval harness
(WER-vs-held-out-day); (c) toy data is a random 15% sample — sessions exist per-trial
(`f[k].attrs["session"]`) but are not arranged into a held-out-day drift split.

---

## Build plan (staged; each phase independently shippable & gated)

### Phase 0 — Repro baseline & drift split *(local, ~½ day)*
- Fix the B0 smoke gate so a clean baseline exists (CE 10.75 is just an unconverged step-50 —
  loosen `profiles/toy.yaml` smoke threshold for the from-scratch BIT case, or seed/warm-up). Record
  baseline toy WER for B0 and G1 to the leaderboard.
- `git submodule add https://github.com/ebrahimfeghhi/transformers_with_dietcorp docks/dietcorp_upstream`,
  pin SHA + `upstream_hash()` in `docks/PINS.txt`. Read their TTA loop as the spec for our native port
  (do **not** wire `run.py` to depend on it).
- **New `tools/drift_eval.py`:** group the val set by `session`/day into an ordered held-out-day
  sequence; provide a "safe-path" synthetic drift generator (per-day affine + Gaussian perturbation
  of real trials, preserving noise stats — the HTML Decision-3 method) so we can run a clean
  N-vs-WER-vs-day curve on toy data without needing 8 real recording days.
- **Gate:** B0 + G1 produce finite WER on toy; drift split yields ≥4 ordered "days".

### Phase 1 — DietCorp TTA loop, native (N=1) *(local, ~1 day)* — your goal (1)
- **New `adapt/dietcorp_tta.py` → `TTAConsolidator`** (TTA is a *between-trial procedure*, not a
  `Stack` forward stage, so it lives outside the encoder→…→decoder chain):
  - `augment(trial, k=64)` — time-masking (~53%) matching DietCorp.
  - `pseudo_label(model, trial)` — self-training label via the model's own greedy/beam decode
    (Qwen is already a strong LM; optional KenLM 3-gram later). **Confidence-gated.**
  - `consolidate(model, trial, n_steps=1)` — N AdamW steps on a **restricted parameter set**
    (default: `dietcorp_recal` per-day affine `day_scale/day_shift` + `proj` — the patch-embed
    analog; configurable to encoder read-in). CTC loss on pseudo-labels (reuse existing
    `stages/loss/ctc_anneal` CTC head).
  - Measures: wake forward latency (must be N-independent) + consolidation wall-clock (≈ linear in N).
- `specs/G2_dietcorp_tta.yaml` + registry `G2` (state: partial). Extend `run.py` with an optional
  `--adapt` eval mode that calls `drift_eval` + `TTAConsolidator` between days.
- **Gate:** with N=1, TTA flattens the synthetic-drift WER curve vs. no-adaptation (reproduces
  DietCorp's qualitative result on toy).

### Phase 2 — Sleep consolidation: sweep N *(local, ~1 day)* — your goal (3)
- Generalize `consolidate(..., n_steps=N)` to N ∈ {1,2,4,8} (N=1 == Phase-1 DietCorp baseline).
- `specs/G3_sleep_consolidation.yaml` (N sweep) + registry `G3`.
- **Primary result:** WER-vs-held-out-day curve per N + wake-latency (flat) + consolidation-cost (∝N).
- **Gate / kill criterion for H_main:** if WER@last-day is **monotonically non-increasing in N**
  with wake latency flat → thesis supported on toy; promote to cloud. If N has no effect →
  documented negative result (still publishable per HTML §Q3).

### Phase 3 — ZenBrain memory, backprop-live *(local, ~1–2 days)* — your goal (2)
- `stages/memory/episodic_buffer.py`: implement **session-keyed + confidence-gated** `EpisodicWritePolicy`;
  make `forward()` real (write recent high-confidence latents, cross-attn read, fuse); set outputs
  `memory_query` / `memory_retrieved`.
- `stages/loss/episodic_consistency.py`: delete the `* 0.0` → live "recall-the-past-latent" objective.
- Flip `registry.yaml` H1 `state: skeleton → partial`, update `.health.json`, and
  **update `tests/test_zenbrain_stub_tripwire.py`** to the inverse contract (asserts H1 is now live,
  read path backprops, loss is non-zero) — the tripwire is *meant* to be flipped, not deleted.
- `specs/H2_zenbrain_live.yaml` + registry `H2`.
- **Gate:** end-to-end backprop through memory stage on toy; episodic loss > 0 and decreasing.

### Phase 4 — Sleep × Memory: ZenBrain replay scheduler *(local→cloud, ~2 days)*
- Combine: the episodic buffer becomes the **replay source** for a between-session sleep pass;
  selection priority = `|TD/CTC-surprise| + confidence + novelty` (ZenBrain Simulation-Selection,
  simplified). Optional: the *learned* consolidation rule (backprop through the N-loop
  consolidate→predict graph) as the most-faithful sleep-paper variant — flagged higher-risk.
- **Gate:** replay-scheduled consolidation ≥ uniform-replay at equal compute on toy drift.

### Phase 5 — Cloud headline *(A100/JarvisLabs, gated)* — eval-scope option B if chosen
- Real Brain-to-Text Benchmark '24, true 8-held-out-day WER-vs-day curve: no-TTA vs DietCorp(N=1)
  vs sleep(N>1) vs sleep+ZenBrain-replay. Enforced by `run.py`'s existing `toy_passed_recently` gate
  before any cloud spend; auto-pause on finish.

---

## Files to create / modify

**Create:** `adapt/__init__.py`, `adapt/dietcorp_tta.py` (TTAConsolidator), `tools/drift_eval.py`
(day-split + synthetic drift + WER-vs-day), `specs/G2_dietcorp_tta.yaml`, `specs/G3_sleep_consolidation.yaml`,
`specs/H2_zenbrain_live.yaml`, `tests/test_dietcorp_tta.py`, `tests/test_drift_eval.py`.

**Modify:** `run.py` (add `--adapt` eval mode invoking drift_eval + TTAConsolidator),
`stages/memory/episodic_buffer.py` (live policy+forward), `stages/loss/episodic_consistency.py`
(remove `* 0.0`), `tests/test_zenbrain_stub_tripwire.py` (invert to live-contract),
`registry.yaml` (add G2/G3/H2; H1 skeleton→partial), `docks/PINS.txt` +
`tools/dietcorp_paper_oracle.yaml` (fill after submodule), `profiles/toy.yaml` (smoke threshold fix).

**Reuse (don't reinvent):** `Stack.from_spec` / `compose_from_spec`; CTC head in
`stages/loss/ctc_anneal.py`; `Preprocessed_BCI_Dataset`, `bci_collate_fn`, `calculate_wer/cer`
via `docks/multiarch_dock.py`; `results/leaderboard.record_run`; `docks/stub.guard_stub_forward`
pattern; `dietcorp_recal` per-day affine as the TTA target params.

---

## Verification

- **Unit (CPU, seconds):** `py -3 -m pytest tests/test_dietcorp_tta.py tests/test_drift_eval.py
  tests/test_zenbrain_stub_tripwire.py -v` — TTA does N grad steps on the restricted param set,
  pseudo-label/confidence gating works, drift split is ordered, memory read path backprops, episodic loss > 0.
- **Smoke:** `py -3 -c "from docks.dietcorp_dock import run_smoke_iters; print(run_smoke_iters(2))"`.
- **Mechanism (local toy, GPU):** `py -3 run.py --expt G3 --profile toy --adapt --train_h5 data/toy_train.hdf5
  --val_h5 data/toy_val.hdf5` → emits WER-vs-day curve per N + wake-latency table to `results/runs/…`.
- **Read the curve:** H_main supported iff WER@last-day decreases with N while wake latency stays flat.

## Risks
- **N-step TTA instability** (sleep paper §7 flags this) → cap N≤8, grad-clip, optimizer-on-subset only.
- **Toy drift is synthetic** → conclusions are qualitative until Phase 5 on real days (state explicitly).
- **Upstream repo may not run on Windows/6 GB** → why it's reference-only, not a dependency.
- **Pseudo-label collapse** → confidence gating + keep core LM frozen (DietCorp updates only patch-embed).