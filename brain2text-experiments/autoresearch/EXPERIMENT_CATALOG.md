# Experiment Catalog — Tracks A, B, C, D, E

**Scope:** the experiments the A100 autoresearch sweep runs — **Tracks A, B, C, D, E**, all on the
A100, all **short toy runs** (no full / 150-epoch runs). Tracks F/G/H (JEPA / DietCorp-TTA /
ZenBrain-episodic) are the separate thesis line and not part of this sweep.

**Baseline:** WER 0.3673 (BIT encoder + MLP projector + Qwen2.5-1.5B + QLoRA).
**Goal:** find the single best building block per stage (encoder / decoder / loss / projector) at
toy scale. **Ranking on toy:** slope (epoch 2→20) + WER@10, vs the track baseline. Long-term full
training of the winning combination is a **separate** downstream goal — not part of this sweep.

---

## Track A — Pretraining Modality (analysis; A4 is E2E)

*Question: why does an audio-pretrained LLM beat a text-only LLM at neural decoding — and is the
benefit about speech specifically, or generic temporal structure?*

| ID | Experiment | Why we run it (the question) | How it might help | Inspiration |
|----|-----------|------------------------------|-------------------|-------------|
| **A1** | CKA Embedding Alignment | Which LLM backbone's embedding space is geometrically closest to the BIT encoder output? | If the audio-LLM aligns best, the projector has a shorter distance to bridge → faster training, lower WER. Tells us which backbone to commit to *before* spending training compute. | **CKA** — Kornblith et al. 2019; audio-backbone gain — Zhang et al. 2025 |
| **A2** | Spoken-Language Perplexity | Is the LLM better calibrated for *spoken* sentences than written text? | Audio pretraining may align the LLM with spoken-sentence statistics (shorter, repetitive) → better language prior for BCI utterances. | Spoken/written LM calibration; Aero1-Audio (Zhang et al. 2025) |
| **A3** | Phoneme Probing | Do frozen LLM hidden states *linearly* encode phoneme identity? | If audio-LLM states are more phoneme-decodable, audio pretraining injects a usable phoneme prior the projector can exploit. | **Linear probing** — Alain & Bengio 2016; diagnostic classifiers |
| **A4** | Audio vs Vision E2E | Is the WER benefit speech-specific, or does any rich-pretraining modality help equally? Swaps only the **LLM decoder backbone** (text Qwen vs audio Qwen2-Audio vs multimodal Phi-4-MM) and measures E2E WER. | Separates a speech prior (H1/H2) from a generic temporal/multimodal prior (H4). Decides which backbone to chase. | LLM modality-pretraining comparison — Qwen2-Audio (Chu 2024) vs Phi-4-MM (Microsoft 2025). *The JEPA latent-alignment version of this question is Track F.* |

> **A4 ↔ Track C overlap:** A4's decoder variants (text / Qwen2-Audio / Phi-4-MM) are the same models as C1/C2. A4 is the *controlled-comparison* framing (WER delta across modalities); Track C runs each decoder as its own experiment. To avoid training the 7B models twice, derive A4's comparison from the C1/C2 leaderboard rows + a text-only baseline rather than re-running them.
>
> **A4 ↔ Track F (JEPA):** A4 answers "audio vs vision" *pragmatically* using off-the-shelf pretrained LLM decoders. The *mechanistic* version — JEPA encoder backbones pretrained under audio-style vs video-style vs neural-style masking — is **Track F** (F1/F2/F3, now active — real wav2vec2-1D / DINOv2-2D / native backbones). JEPA is the conceptual inspiration for the question, not a component of A4.

---

## Track B — Encoder Architecture

*Question: which neural encoder extracts the most speech-decodable representation? Projector +
decoder held fixed (MLP + Qwen2.5-1.5B). All judged against B0 (BIT from scratch, no SSL).*

| ID | Experiment | Why we run it (the question) | How it might help | Inspiration |
|----|-----------|------------------------------|-------------------|-------------|
| **B0** | BIT Baseline (from scratch) | The fair control — BIT with no SSL checkpoint, so B1–B5 (which also lack SSL) are compared honestly. | Establishes the slope/WER@10 every other encoder must beat. | **BIT** — Zhang et al. 2025 |
| **B1** | ConformerXL | Does correcting neural spike-timing jitter *before* patching produce cleaner tokens? | Dilated-conv + BiGRU prenet smooths ~10 ms firing jitter → cleaner patches → lower PER → lower WER. | **ConformerXL/iPhoneme** ALS paper; Conformer — Gulati et al. 2020 |
| **B2** | HRM Dual-Timescale | Does processing at 20 ms *and* 100 ms (hierarchical, DEQ fixed-point) beat a flat transformer? | Mirrors cortical hierarchy; the L-module settles per 20 ms bin before the H-module integrates → better temporal features at O(1) memory. | **HRM** (Hierarchical Reasoning Model) |
| **B3** | MambaPOSSM (GRU fallback) | Does a selective state-space model beat attention for sparse spike trains? | The SSM learns *which* channels to track over time (input-dependent state) → compact, causal, O(L) memory. The cross-attention tokenizer also replaces a hard reshape with learned selection. | **Mamba** — Gu & Dao 2023; **POSSM** hybrid decoding |
| **B3_mamba** | MambaPOSSM (true SSM) *(A100)* | Same as B3 but with the real CUDA Mamba kernel (won't build on Windows). | Confirms whether the true selective-scan outperforms the GRU proxy. | Mamba — Gu & Dao 2023 |
| **B4** | MoE Encoder | Does routing tokens to specialized experts capture the multi-domain nature of neural signals? | 6 routed + 2 shared experts let different patches (phoneme / prosody / motor phases) use specialized FFNs → richer features. | **EEGMoE**; **NeuroMoE**; Sparse MoE — Shazeer et al. 2017 |
| **B5** | ZenBrain Memory | Does cross-attention over an episodic buffer of high-confidence past trials help noisy ones? | Clean cached patch embeddings guide decoding of noisier later trials → zero-calibration robustness to day-to-day drift. | **ZenBrain** memory architecture; **DietCORP** time-masked TTA |

---

## Track C — LLM Decoder Variants

*Question: does an audio/multimodal-pretrained LLM decoder bridge the neural→text gap better than
text-only Qwen2.5-1.5B? BIT encoder + MLP projector held fixed; only the decoder LLM changes. All
three fit the A100 (40 GB), so the whole track is in scope.*

| ID | Experiment | Why we run it (the question) | How it might help | Inspiration |
|----|-----------|------------------------------|-------------------|-------------|
| **C1** | Qwen2-Audio-7B | Does an audio-pretrained 7B LLM (used as a text decoder via `inputs_embeds`) beat text-only Qwen? | Audio pretraining may give the decoder a speech/phoneme prior that reads neural embeddings more readily — the closest analogue to the paper's Aero1-Audio backbone. | **Qwen2-Audio** (Chu et al. 2024); Aero-1-Audio (Zhang 2025) |
| **C2** | Phi-4-Multimodal | Does a speech+vision pretrained 5.6B decoder help — both acoustic and temporal priors? | Multimodal pretraining (Azure speech + video) may add both a speech prior and a temporal-sequence prior at smaller size than C1. | **Phi-4-Multimodal** (Microsoft 2025) |
| **C3** | Whisper-Qwen Split Stack | Does passing neural features through a *frozen* Whisper speech manifold before Qwen help? | Whisper's learned speech manifold may act as a "normalizer," mapping neural patterns into a space the LLM was trained to read — the most principled test of the embedding-geometry hypothesis (H3). | **Whisper** (Radford et al. 2022); speech-LLM bridging (SALMONN) |

---

## Track D — Loss Function Design

*Question: how should CE + CTC + contrastive be combined and scheduled? Same architecture (BIT +
MLP + Qwen) throughout, so differences isolate the loss term.*

| ID | Experiment | Why we run it (the question) | How it might help | Inspiration |
|----|-----------|------------------------------|-------------------|-------------|
| **D1b** | CTC Linear Anneal | Should CTC be strong early (force phoneme learning) and fade late? | Early CTC organizes encoder time-tokens phonemically; annealing to 0 by epoch 75 frees the encoder to specialize for E2E decoding. | **CTC** — Graves et al. 2006; curriculum learning |
| **D1d** | CTC Ablation (weight 0) | Does CTC help *at all* in E2E, or is it redundant with CE? | If removing CTC doesn't hurt WER, the loss simplifies. A clean negative result either way. | CTC — Graves et al. 2006 |
| **D2a** | Contrastive Ablation (weight 0) | Is the InfoNCE contrastive loss helping, or *conflicting* with CE? | If weight=0 *improves* WER, the contrastive term was pushing a bad local minimum → remove or reschedule it. | **InfoNCE** — van den Oord 2018; CLIP — Radford et al. 2021 |
| **D2d** | Contrastive ×2 | Does *stronger* neural↔text alignment lower WER? | Tighter modality alignment → projector lands neural pooled embeddings nearer their sentence embeddings. | InfoNCE; CLIP |
| **D3b** | TopoLoss (λ=0.001) | Does light cortical-map regularization on FFN weights improve generalization? | Forcing nearby FFN neurons to be functionally similar → robustness to perturbation, better generalization on ~1000 samples. Zero new parameters. | **TopoNets** — high-performing topographic models |
| **D3c** | TopoLoss (λ=0.01) | Standard-strength topographic regularization — find the sweet spot. | Same mechanism, stronger; D3b/D3c bracket the useful λ range. | **TopoNets** |
| **D4** | Label Smoothing (ε=0.1) | Does smoothing prevent overfitting on the tiny BCI sentence set? | Stops the model becoming overconfident on exact training sentences → better val generalization. | **Label smoothing** — Szegedy et al. 2016; Müller et al. 2019 |

---

## Track E — Projector Architecture

*Question: the projector does a 384→1536 dimensionality expansion AND a neural→text semantic shift.
Is it the underexplored bottleneck? Encoder (BIT) + decoder (Qwen) held fixed.*

| ID | Experiment | Why we run it (the question) | How it might help | Inspiration |
|----|-----------|------------------------------|-------------------|-------------|
| **E1a** | Deep MLP (5-layer) | Is projector *depth* the bottleneck in the semantic bridge? | More layers → richer neural→text transformation than the 3-layer default. | MLP scaling; LLaVA-style projectors (Liu et al. 2023) |
| **E1b** | Gated MLP | Does multiplicative gating beat plain ReLU MLP? | `relu(fc1(x)) * sigmoid(gate(x))` adds input-dependent feature selection. | **GLU/Gated MLP** — Dauphin et al. 2017; Shazeer 2020 |
| **E2b** | Q-Former (32 queries) | Does cross-attention with learned queries beat a per-token MLP projector? | 32 learned queries attend to the *full* neural sequence → fixed-length output, no padding waste, queries focus on phoneme-salient windows. *Highest-impact projector change.* | **Q-Former / BLIP-2** — Li et al. 2023 |
| **E3** | Patch × Query Grid | What patch_size × n_queries trade-off is optimal? | 2×3 grid balances temporal resolution (patch) vs LLM sequence length (queries). | BLIP-2 + patch tokenization (ViT — Dosovitskiy et al. 2020) |

---

## Track F — JEPA Self-Supervised Pretraining (audio vs video vs neural)

*Question: the mechanistic version of A4. Instead of borrowing an off-the-shelf pretrained LLM,
pretrain our **own** encoder backbone with a Joint-Embedding Predictive objective under different
modality "lenses," and ask which latent structure the neural signal aligns with best. Controlled
A/B/C: the F1/F2/F3 specs are **byte-identical except one `modality:` line** (`tools/diff_specs.py`
enforces this). Produces a pretrained backbone, **not a WER** — ranking is on pretraining health
(no collapse) + downstream decoding once fine-tuned.*

> **Status: ACTIVE.** F runs in this sweep. `stages/encoder/jepa.py` now implements real
> per-modality backbones — a **wav2vec2-style 1D temporal conv stack** (audio), a **DINOv2-style 2D
> patch conv** (video), and a **native patch-embed** (neural control) — trained via the real JEPA
> objective (EMA target + stop-grad + masked-latent prediction + VICReg anti-collapse). The
> controlled A/B/C holds: F1/F2/F3 stay byte-identical except `modality:` (enforced by
> `tools/diff_specs.py`), and all three emit the identical `(B, T_patch, 384)` contract, so only the
> inductive bias differs. Pretrained audio/image *weights* are not used — the input is neural
> (B,T,512), so only the architectural bias transfers. F is a *pretraining* track: it yields a
> backbone + downstream eval, not a single toy WER. *(Smoke-verified: 4/4 `tests/test_jepa_smoke.py`
> pass — shape contract, loss-decrease, stop-gradient, spec-identity.)*

| ID | Experiment | Why we run it (the question) | How it might help | Inspiration |
|----|-----------|------------------------------|-------------------|-------------|
| **F1** | Audio-JEPA Backbone | Does an audio-style masked-latent objective yield a backbone whose latents decode neural signals well? | Confirms (or refutes) a brain↔audio latent affinity at the *representation* level — deeper than A4's weight-init test. | **I-JEPA** (Assran et al. 2023); **VICReg** (Bardes et al. 2022); wav2vec2 (Baevski 2020) |
| **F2** | Video-JEPA Backbone | Same objective, video-style patchifier — does visual/temporal masking align better or worse than audio? | Controlled contrast to F1: only the patchifier (modality) changes, so any gap is attributable to modality. | **V-JEPA** (Bardes et al. 2024); DINOv2 (Oquab et al. 2023) |
| **F3** | Neural-JEPA Backbone *(provocation arm)* | Does JEPA-pretraining **directly on the BCI signal** beat borrowing audio/visual structure? | If neural-JEPA wins, the signal is best modeled on its own terms — a 3-way controlled test of the modality question. | **JEPA** — LeCun 2022 ("A Path Towards Autonomous Machine Intelligence") |

*Dependencies: F2 and F3 both depend on F1. Integrity gate: `tools/diff_specs.py` must confirm the
three specs differ only in the `modality:` line; smoke test `tests/test_jepa_smoke.py`.*

---

## Combination phase (after single-lever sweeps)

Compose the **best encoder (B) × best decoder (C) × best loss (D) × best projector (E)** in one
toy run via `--override`. If the combination's slope beats each ingredient alone, that is the
best-building-block stack to hand to the **separate** long-term-implementation goal; if it
underperforms, the levers interact negatively — report it and carry forward the best single lever.
(Still toy-only — no full run is executed here.)

| Run | Composition | Why |
|-----|-------------|-----|
| **CMB-1** | Best-B encoder + best-C decoder + E2b Q-Former + best-D loss | Test whether all four winners stack (additive-gains hypothesis) |
| **CMB-2** | Best-B encoder + best-E projector + D1b anneal + D2a (drop contrastive if D2a won) | Keep the architectural winners but simplify the loss (loss-simplification hypothesis) |

---

## Summary counts

All runs on the A100, all `--profile toy` (short, diagnostic — no full runs).

| Track | Experiments | Count |
|-------|-------------|-------|
| A | A1, A2, A3, A4 | 4 |
| B | B0, B1, B2, B3, B3_mamba, B4, B5 | 7 |
| C | C1, C2, C3 | 3 |
| D | D1b, D1d, D2a, D2d, D3b, D3c, D4 | 7 |
| E | E1a, E1b, E2b, E3 | 4 |
| F | F1, F2, F3 *(JEPA pretraining — controlled A/B/C)* | 3 |
| **Total** | A / B / C / D / E / F | **28** (+ 2 combination runs) |

*Tracks G (DietCorp-TTA) and H (ZenBrain-episodic) remain the separate thesis line — not in this
sweep. Track F is ACTIVE: real wav2vec2-1D / DINOv2-2D / native backbones in
`stages/encoder/jepa.py`, smoke-verified, run as a controlled A/B/C pretraining campaign.*
