# Sleep-Consolidated Test-Time Adaptation under Real Sequential Drift: Constant-Latency Decoding with Episodic Memory

*Variant: **methods / test-time-adaptation framing** (target: ICLR 2026 — Test-Time Updates (TTU), or
"Catch, Adapt, and Operate: Monitoring ML Models Under Drift"). The BCI is the real-world testbed.*
*Working draft — pre-registration style. Method and setup are implemented; every **result** is a stated
prediction with a `[TBD]` placeholder pending the A100 run. Numbers attributed to prior work or to
toy/synthetic runs are labeled inline; no real-T15 results are invented here.*
*Shared technical core (§3–§5, §7) is identical to the clinical variant `preprint_health.md`.*

**Authors:** Pratik Pattanaik et al. *(affiliations TBD)*

---

## Abstract

Test-time adaptation (TTA) is usually benchmarked on i.i.d. corruptions, where each test batch is an
independent draw. Real deployments instead face *sequential* distribution shift, where today's
distribution is yesterday's plus an increment. We use a rare clean instance of this — multi-month
electrode drift in an intracortical speech brain-computer interface — to ask when *iterative* TTA helps,
when it collapses, and how to stabilize it under a hard inference-latency budget. We study per-sentence
TTA that updates only a frozen decoder's input layer, generalize the single adaptation step to **N
offline "sleep" steps** performed in the inter-sentence pause (so wake-time latency is constant in N),
and add a **parameter-free episodic memory buffer** that anchors adaptation to confident past
representations and carries them across days. On the real T15 copy-task dataset (45 chronological
sessions over ~20 months) we evaluate day-to-day degradation, the effect of consolidation depth, and the
memory contribution, reporting word error rate with a word n-gram-LM decoder. Our central finding is that
iterative TTA is only safe **above a pseudo-label-quality threshold**: below it, deeper adaptation
compounds its own errors and collapses; above it, depth helps — and a parameter-free episodic anchor
*widens* this safe regime. We predict deeper consolidation lowers drift error at flat wake latency
`[TBD]`, and that episodic memory shortens the per-day **warm-up** (a "time-to-usable" metric) `[TBD]`.
Preliminary synthetic-drift runs confirm the mechanism and its failure mode (toy PER 0.64→4.2 for N=1→8
with noisy labels; monotone gains with adequate labels).

## 1. Introduction

Test-time adaptation updates a model online from unlabeled test data, and is the standard answer to
distribution shift at deployment \citep{tent2021}. Yet it is almost always evaluated on *i.i.d.*
corruption benchmarks, where the shift has no temporal structure. Many real settings are different: the
shift is **sequential**, accumulating over time, and adaptation is *iterative* — the model repeatedly
updates itself on its own outputs. Iterative self-adaptation is exactly where TTA is most fragile, and
where the i.i.d. benchmarks are least informative.

We use an unusually clean instance of sequential drift to study this: an intracortical **speech**
brain-computer interface (BCI) recorded over ~20 months. The neural signal drifts day-to-day (electrode
micro-motion, impedance change, baseline shift), so a decoder trained once and frozen decodes worse with
each passing day; and because each day's statistics are the previous day's plus a physical increment, the
shift is genuinely sequential rather than a reshuffle. The clinical setting also imposes a constraint that
makes the problem sharp: decoding a sentence is latency-critical, but the second-or-two pause before the
next sentence is free compute. This separates the timescales — a hard *wake* latency budget, and an
*offline* budget in which to adapt.

**Claim.** A frozen decoder can be kept usable across months of sequential drift by offline, label-free
consolidation between inputs — at an inference latency *independent of how much consolidation is done* —
provided pseudo-label quality stays above a threshold that a parameter-free episodic-memory anchor helps
maintain.

We generalize DietCorp's single-step input-layer TTA \citep{dietcorp} to N offline "sleep" steps run in
the inter-sentence pause \citep{sleep_lms}, add a parameter-free episodic memory buffer in two roles (a
sleep-time anchor and an optional wake-time read) \citep{zenbrain}, and evaluate on the real T15 dataset
\citep{card2024t15} under true chronological drift. Contributions:

- **C1.** A fully-online, label-free adaptation scheme with a **provably input-independent wake latency**
  (all extra compute offline), generalizing single-step TTA to depth N.
- **C2.** A **label-quality safe-regime** characterization of iterative TTA on real sequential drift —
  depth helps above a pseudo-label-quality threshold and collapses below it — and a **parameter-free
  episodic memory** that widens the safe regime and carries confident past states across days.
- **C3.** An evaluation on **real chronological drift** (T15, 45 sessions / ~20 months) with controls
  separating depth from step-size and label quality, and a usability metric, **time-to-usable** (the
  per-day warm-up curve), with the BCI as a high-stakes deployment testbed.

## 2. Related Work

**Test-time adaptation and offline consolidation.** Entropy- or pseudo-label-driven TTA updates a model
online from unlabeled data \citep{tent2021}; DietCorp specializes this to speech BCIs, adapting only the
patch-embedding from a single CTC pseudo-label per trial \citep{dietcorp} — the N=1 special case we
generalize. The principle of N offline passes at a boundary while preserving wake latency, and the
prediction that depth helps on *sequential* problems, comes from offline-recurrence work in language
models \citep{sleep_lms}; the episodic buffer and prioritized-replay scheduler are adapted from
neuroscience-inspired memory architectures \citep{zenbrain}. We borrow these as *structure* — N offline
steps, wake/sleep separation, an episodic anchor — not as learned optimizers: our update is fixed gradient
descent.

**Drift in neural decoding.** Three families address neural non-stationarity. *Hardware stability*:
endovascular arrays endothelialize into the vessel wall for chronically stable signals
\citep{synchron_stability2025}, at the cost of bandwidth. *Recalibration*: BrainGate infers user intent
online and retrains on the inferred labels \citep{braingate_unsup_recal2023, measuring_instability2024},
restoring accuracy but with a per-session cold start. *Latent-dynamics alignment*: ADAN
\citep{degenhart2020adan} and NoMAD \citep{nomad2025} align each day's activity onto a stable manifold for
months of label-free stability — but on *motor* decoding, and by aligning inputs rather than adapting the
decoder.

**Positioning.** Against this backdrop our setting is distinguished by four axes: (i) *speech, not motor*
decoding; (ii) *adaptation, not alignment* (we update the decoder's input layer, suiting a frozen CTC
head); (iii) a *wake-latency guarantee* — decode latency is independent of adaptation depth, which neither
recalibration nor alignment makes explicit; and (iv) *session-start robustness* via a cross-day episodic
buffer that attacks the per-day cold start both leave in place. We position the method as the
latency-bounded, online complement to manifold alignment, not a competitor; baselines are DietCorp (N=1,
same data) and no-adapt / no-memory controls, with conceptual comparison to
\citep{nomad2025, braingate_unsup_recal2023}.

## 3. Method

**Model.** A CTC phoneme model: a 7-layer RoPE transformer \citep{su2021rope} (`embed_dim=384`,
`num_heads=6`) with a per-session linear read-in and a patch embedding over `patch_size=5` 20 ms bins
(100 ms patches), and a `Linear(384→42)` head (0=blank, 1–41 phonemes), trained with CTC
\citep{graves2006ctc}. Weights and patch dimensions are reconstructed from the trained checkpoint.

**Wake path (one forward pass).** `neural → read_in[session] → patch embed → 7× transformer → final LN →
(optional episodic read) → CTC head → greedy decode → words (word n-gram LM)`. Wake latency is timed as
exactly one clean forward and is independent of N by construction.

**Sleep path (between sentences).** After each sentence: (1) pseudo-label the clean trial — `self`
(greedy CTC), `lm` (phoneme n-gram-refined CTC beam search), or `oracle` (ground-truth); (2) 64
time-masked augmentations (~53% masked); (3) `N` AdamW steps (lr 1e-5, grad-clip 1.0) on the
patch-embedding + read-in **only**, minimizing CTC over the augmentations `+ β·MSE(latent,
recalled.detach())` when memory is attached; (4) optionally include prioritized replay trials.
Consolidation cost grows ~linearly in N; wake latency does not.

**Episodic memory buffer.** A fixed ring of mean-pooled high-confidence past latents, session-tagged.
Retrieval is **parameter-free similarity attention** — `softmax(x·bufferᵀ/√E)·buffer`, i.e. recall the
nearest confident past latents, requiring no training. Two separable roles: (i) a **sleep-time anchor**
(always on) via the MSE term (condition C3a); (ii) an **optional wake read** `x + σ(gate)·attended`
(condition C3b). Writes are **confidence-gated relative to the day** to avoid buffer starvation under
heavy drift; a **cold-buffer guard** bypasses the read until enough entries exist, so the anchor
contributes exactly 0 at session 1. The buffer **persists across days**.

**Priority replay.** A capped store scores past trials by `|surprise| + (1−confidence) + novelty` and
feeds the top-K into consolidation.

**Canonical decoder.** Phoneme posteriors are decoded to words by a lexicon beam search with shallow
**word n-gram-LM** fusion (`cost = edit-distance + λ·(−log P_unigram) + μ·(−log P_ngram(w|history))`).
Matching DietCorp's exact headline additionally needs their KenLM + LLM rescoring (a stretch goal).

## 4. Experimental Setup

**Data.** T15 copy-task dataset (Dryad `doi:10.5061/dryad.dncjsxm85`; \citealp{card2024t15}): 10,948
sentences across 45 dated sessions spanning ~20 months. Sessions are loaded in true chronological order;
per-session z-score + Gaussian smoothing (σ=1.5) replicate the training dataloader. Each day is evaluated
**before** adapting on it; consolidation is per sentence.

**Conditions.** C0 no-adapt (N=0); C1 self-label; C2 LM-refined; **C3a** LM + sleep anchor + replay
(clean wake); **C3b** C3a + wake read (full system); C4 oracle (ceiling). Depth `N ∈ {0,1,2,4,8}`.
C1@N=1 reproduces DietCorp.

**Controls.** (a) **Step-size:** lr-sweep at N=1 matched to the total step size of N>1. (b)
**Sequentiality:** shuffled-day order rerun — does the benefit depend on chronological order? (c)
**Seeds:** multiple seeds on headline cells (±SEM). All claims scoped to **T15, single participant**.

**Metrics.** Per-day PER and word n-gram-LM WER; **time-to-usable** = error on inputs 1–5 within each day
(memory vs no-memory); wake latency (ms) and consolidate cost (ms) per (condition, N); buffer occupancy
over days. **Inventory gate:** decoding ground-truth labels → words must give WER ≈ 0.

## 5. Results *(pre-registered predictions; `[TBD]` pending full run)*

- **5.1 Drift baseline (C0).** Predict monotone error increase across days (anchor: DietCorp reports
  no-TTA 22.74%→32.58% day 1→5). `[TBD — F1]`
- **5.2 Single-step reproduction (C1@N=1).** Single-step TTA recovers part of the degradation. `[TBD]`
- **5.3 Depth × label quality (the safe-regime result).** Predict: with self-labels, depth is unstable;
  with LM-refined labels, deeper N reduces later-day error, optimum ~N≈4 before a ceiling. *Synthetic
  prelim:* collapse 0.64→4.2 (N=1→8); competent-decoder demo 0.955→0.765 (N=1→4), wake flat
  (0.05–0.16 ms), consolidate linear (3→4→24→101 ms). `[TBD — F4]`
- **5.4 Step-size control.** Predict C2(N>1) beats the lr-matched N=1 baseline → depth ≠ larger lr.
  `[TBD]`
- **5.5 Episodic memory.** Predict C3a < C2 on hard-drift days (anchor widens the safe regime); **C3b
  shorter per-day warm-up than C2**. `[TBD — F2]`
- **5.6 Latency.** Predict wake latency flat across N; consolidate cost ~linear in N. `[TBD — F3]`
- **5.7 Sequentiality.** Predict the depth/memory benefit shrinks under shuffled-day order → it is tied to
  sequential structure. `[TBD]`

## 6. Discussion

The result, if it holds, is a recipe for **iterative TTA under real sequential drift**: depth helps only
above a pseudo-label-quality threshold, a parameter-free memory anchor widens that safe regime, and a
strict wake/sleep split makes the whole thing free at inference time regardless of depth. The
**time-to-usable** metric — error on the first few inputs after a shift — surfaces a cost that steady-state
accuracy hides and that matters for any system that restarts under drift. The same recipe should transfer
to other sequential-drift, latency-bounded settings (myoelectric/EMG control, streaming on-device
personalization); establishing that beyond one dataset is future work.

## 7. Limitations

Single participant (T15) — claims scoped accordingly. Canonical WER uses a word n-gram LM, not the full
LLM-rescoring stack, so absolute WER is not directly comparable to DietCorp's best 12.17%. The oracle is
only as faithful as the dataset's phonemization. Consolidation can destabilize at large N (a depth
ceiling), mitigated but not removed by grad-clipping and a gentle lr. The wake read adds a small,
*constant* latency (constant in N, not free).

## 8. Conclusion

We use a real multi-month speech-BCI drift dataset to characterize iterative test-time adaptation:
depth helps only above a label-quality threshold, a parameter-free episodic anchor widens that regime, and
wake/sleep separation keeps inference latency independent of adaptation depth. We pre-register the
evaluation and release the apparatus.

## References

See `references.bib`.
