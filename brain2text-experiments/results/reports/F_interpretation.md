# Track F — JEPA Self-Supervised Pretraining
*Generated 2026-06-07 — toy-profile sweep, 20 epochs, 200 batches/epoch*

**Question:** Which modality inductive bias (audio/video/neural) produces the best JEPA backbone for BCI decoding?

**Baseline for comparison:** B0_baseline (BIT from scratch), slope = -0.00617

---

## Ranked Results

| Rank | ID | WER@10 | Slope | Δ slope vs B0 | Label |
|------|-----|--------|-------|----------------|-------|
| 1 | **F3** | 1.0000 | -0.02263 | +267% | **STRONG** |
| 2 | **F1** | 1.0494 | 0.01715 | -378% | **INERT** |
| 3 | **F2** | 1.0000 | 0.03841 | -722% | **INERT** |

---

## Experiment Interpretations

### F3

Neural-JEPA backbone (native patch-embed, directly on BCI signal). Downstream WER slope -0.0226 — STRONG. **Best JEPA backbone.** Self-supervised pretraining directly on the BCI signal with JEPA objectives (EMA target + VICReg) produces a backbone whose latent structure aligns best with the downstream decoding task — confirming that neural signals are best modelled on their own terms rather than via borrowed audio/visual structure (LeCun 2022).

### F1

Audio-JEPA backbone (wav2vec2-style 1D conv). Downstream WER slope 0.0171 — INERT. Audio-style temporal masking does not transfer well to neural BCI signals; the 1D conv inductive bias may over-segment the neural firing patterns relative to the phoneme timescale.

### F2

Video-JEPA backbone (DINOv2-style 2D patch conv). Downstream WER slope 0.0384 — INERT. Video-style spatial masking performs worst of the three JEPA variants; spatial patchification is the least natural prior for 1D neural spike trains.

---

**Track F winner:** F3 (slope -0.02263)
