# Reliable from the First Sentence: Recalibration-Free, Constant-Latency Adaptation for At-Home Speech Neuroprostheses under Electrode Drift

*Variant: **clinical / ML-for-health framing** (target: a NeurIPS-style health venue — Learning from Time
Series for Health (TS4H) or Machine Learning for Health (ML4H), or a clinical-translation track).*
*Working draft — pre-registration style. Method and setup are implemented; every **result** is a stated
prediction with a `[TBD]` placeholder pending the A100 run. Numbers attributed to prior work or to
toy/synthetic runs are labeled inline; no real-T15 results are invented here.*
*Shared technical core (§3–§5, §7) is identical to the methods variant `preprint_ttu.md`.*

**Authors:** Pratik Pattanaik et al. *(affiliations TBD)*

---

## Abstract

For people who have lost the ability to speak — through ALS, brainstem stroke, or locked-in syndrome — an
intracortical speech neuroprosthesis can restore communication, but only if it stays usable day after day
in the home, without a technician. The core obstacle is electrode drift: the neural signal changes over
weeks and months, so a decoder trained once and frozen degrades, and current systems recover by
recalibrating each session — which interrupts use and leaves every morning with a cold start, precisely
when a patient may most need to communicate. We present a decoder-side scheme that keeps a frozen speech
decoder usable online with **no recalibration blocks** and **no increase in response latency**:
per-sentence test-time adaptation that updates only the input layer, generalized to **N offline "sleep"
steps** run in the natural pause between sentences, plus a **parameter-free episodic memory** that carries
the patient's confident past neural states across days. On the real T15 copy-task dataset (45 sessions
over ~20 months) we evaluate day-to-day degradation, the effect of consolidation depth, and the memory
contribution, reporting word error rate. The clinically meaningful outcome we test is **time-to-usable**:
episodic memory should shrink the per-day warm-up so the device is reliable from the first sentence each
morning `[TBD]`, at a response latency that never grows with adaptation `[TBD]`. We also characterize when
online adaptation is **safe for unsupervised use** — it requires sufficiently reliable pseudo-labels, and
the memory anchor broadens that safe operating range. Preliminary synthetic-drift runs confirm the
mechanism and its failure mode (toy PER 0.64→4.2 for N=1→8 with noisy labels; monotone gains with
adequate labels).

## 1. Introduction

A speech neuroprosthesis is, for some patients, the only viable channel to communicate with family and
caregivers \citep{card2024t15, willett2023speech}. Recent systems decode attempted speech at low error
rates on the day they are calibrated. But the value to a patient depends not on peak accuracy in a lab
session — it depends on whether the device works **reliably, every day, at home, without expert
intervention**. That is where electrode drift bites: the recorded signal is non-stationary, accumulating
change from micro-motion, impedance shifts, and baseline drift, so a frozen decoder degrades over days and
weeks. The standard remedy is to recalibrate each session, which requires supervision, interrupts use, and
imposes a daily **cold start** — the first ten-to-twenty sentences of every session are unreliable. For a
patient who wakes and needs to communicate a medical need, "reliable after two minutes" and "reliable
immediately" are very different devices.

We target deployability directly: **can a frozen speech decoder be kept usable online — without
recalibration blocks, and without ever slowing the patient's responses — across months of drift?** The
clinical workflow makes this possible. Decoding a sentence must be fast, but the natural one-to-two second
pause before the patient speaks again is free compute. We spend that pause adapting the decoder, so the
patient never waits longer for deeper adaptation, and we use an episodic memory of the patient's own
confident past neural states to make each new day start warm rather than cold.

**Claim.** A frozen speech-BCI decoder can be kept usable across months of drift by offline, label-free
consolidation between sentences — at a response latency *independent of how much consolidation is done* —
so that the device is reliable from the first sentence each session, without any technician recalibration.

We generalize DietCorp's single-step input-layer adaptation \citep{dietcorp} to N offline "sleep" steps
in the inter-sentence pause \citep{sleep_lms}, add a parameter-free episodic memory in two roles (a
sleep-time anchor and an optional wake-time read) \citep{zenbrain}, and evaluate on the real T15 dataset
\citep{card2024t15} under true day-to-day drift. Contributions:

- **C1.** A **recalibration-free, fully-online** adaptation scheme for speech BCIs with a **constant
  response latency** regardless of adaptation depth (all extra compute is offline, between sentences).
- **C2.** A parameter-free **episodic memory** that carries the patient's confident past states across
  days, directly targeting the daily **cold start** that recalibration leaves in place — and a
  characterization of when unsupervised online adaptation is *safe* (it requires adequate pseudo-label
  quality, which the memory anchor helps maintain).
- **C3.** An evaluation on **real chronological drift** (T15, 45 sessions / ~20 months) reporting a
  clinically meaningful **time-to-usable** metric (per-day warm-up), alongside standard word error rate.

## 2. Related Work

Deployed and investigational BCI programs address neural non-stationarity in three ways, each leaving a
usability gap for at-home speech.

**Hardware stability.** Endovascular arrays (Synchron's Stentrode) endothelialize into the vessel wall and
report chronically stable signals with little drift \citep{synchron_stability2025} — an elegant solution,
but with far fewer channels than high-rate speech decoding needs.

**Recalibration.** The dominant intracortical approach is to retrain the decoder; BrainGate made this
unsupervised by inferring the patient's intent and retraining on inferred labels
\citep{braingate_unsup_recal2023}, and quantifying the instability that necessitates it is an active topic
\citep{measuring_instability2024}. This restores accuracy but keeps a per-session cold start and
adaptation overhead — exactly the burdens a home device should remove.

**Latent-dynamics alignment.** ADAN \citep{degenhart2020adan} and NoMAD \citep{nomad2025} align each day's
activity onto a stable manifold for months of label-free stability — impressive, but demonstrated for
*motor/cursor* control, and still cold at the start of each session.

**Positioning.** Our contribution is the clinically-motivated software layer these leave open: it targets
**speech** rather than motor decoding; it **adapts** the decoder online rather than aligning a manifold;
it guarantees a **constant response latency** (no method above frames adaptation in patient-latency terms);
and it explicitly attacks **session-start reliability** by carrying the patient's confident past states
across days. Being decoder-side software, it is **hardware-agnostic** and could ride on any chronic speech
BCI. Baselines are DietCorp (N=1, same data) and no-adapt / no-memory controls, with conceptual comparison
to \citep{nomad2025, braingate_unsup_recal2023}.

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
contributes exactly 0 at session 1. The buffer **persists across days**, carrying yesterday's best states
into today's warm-up.

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
(clean wake); **C3b** C3a + wake read (full device); C4 oracle (ceiling). Depth `N ∈ {0,1,2,4,8}`.
C1@N=1 reproduces DietCorp.

**Controls.** (a) **Step-size:** lr-sweep at N=1 matched to the total step size of N>1. (b)
**Sequentiality:** shuffled-day order rerun. (c) **Seeds:** multiple seeds on headline cells (±SEM). All
claims scoped to **T15, single participant**.

**Metrics.** Per-day PER and word n-gram-LM WER; **time-to-usable** = WER on sentences 1–5 within each day
(memory vs no-memory); wake latency (ms) and consolidate cost (ms) per (condition, N); buffer occupancy
over days. **Inventory gate:** decoding ground-truth `seq_class_ids → words` must give WER ≈ 0.

## 5. Results *(pre-registered predictions; `[TBD]` pending full run)*

- **5.1 Drift baseline (C0).** Predict monotone WER increase across days (anchor: DietCorp reports no-TTA
  22.74%→32.58% day 1→5). `[TBD — F1]`
- **5.2 DietCorp reproduction (C1@N=1).** Single-step TTA recovers part of the degradation. `[TBD]`
- **5.3 Depth × label quality.** Predict: with self-labels, depth is unstable; with LM-refined labels,
  deeper N reduces later-day WER, optimum ~N≈4 before a ceiling. *Synthetic prelim:* collapse 0.64→4.2
  (N=1→8); competent-decoder demo 0.955→0.765 (N=1→4), wake flat (0.05–0.16 ms), consolidate linear
  (3→4→24→101 ms). `[TBD — F4]`
- **5.4 Step-size control.** Predict C2(N>1) beats the lr-matched N=1 baseline. `[TBD]`
- **5.5 Episodic memory (load-bearing, clinical).** Predict C3a < C2 on hard-drift days; **C3b shows a
  shorter per-day warm-up than C2** — the device is reliable from the first sentence. `[TBD — F2, headline]`
- **5.6 Latency.** Predict wake latency flat across N; consolidate cost ~linear in N — deeper adaptation
  never slows the patient. `[TBD — F3]`
- **5.7 Ceiling (C4).** Oracle labels upper-bound WER; the gap attributes remaining error to label
  quality. `[TBD]`

## 6. Significance

For the speech-BCI population (ALS, brainstem stroke, locked-in syndrome), the contribution is not a new
accuracy record but a step toward a **practical home device**: a frozen decoder kept usable day-to-day
**without technician recalibration**, that **never slows the patient's responses** as it adapts, and that
is **reliable from the first sentence each morning**. Because it is decoder-side software, it is
**hardware-agnostic** and could be adopted by any chronic speech-BCI program, complementing
stability-by-design hardware. The nearest transfer is other assistive biosignal interfaces with
session-to-session electrode-shift drift (myoelectric prosthetics). The **time-to-usable** metric we
introduce captures the patient-facing cost that steady-state accuracy hides. Validation beyond a single
participant is the necessary next step before clinical claims.

## 7. Limitations

Single participant (T15) — claims scoped accordingly; no clinical efficacy is claimed. Canonical WER uses
a word n-gram LM, not the full LLM-rescoring stack, so absolute WER is not directly comparable to
DietCorp's best 12.17%. The oracle is only as faithful as the dataset's phonemization. Consolidation can
destabilize at large N (a depth ceiling), mitigated but not removed by grad-clipping and a gentle lr. The
wake read adds a small, *constant* latency (constant in N, not free).

## 8. Conclusion

We present and pre-register the evaluation of a recalibration-free, constant-latency, drift-robust
adaptation scheme for at-home speech neuroprostheses, combining offline consolidation with an episodic
memory that carries a patient's confident representations across days. The central claim — *reliable from
sentence one, every day, without recalibration* — is what would make such a device practical, and is what
the real-T15 experiments are designed to test.

## References

See `references.bib`.
