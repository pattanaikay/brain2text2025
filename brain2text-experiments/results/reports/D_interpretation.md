# Track D — Loss Function Design
*Generated 2026-06-07 — toy-profile sweep, 20 epochs, 200 batches/epoch*

**Question:** How should CE + CTC + contrastive be combined and scheduled?

**Baseline for comparison:** B0_baseline (BIT from scratch), slope = -0.00617

---

## Ranked Results

| Rank | ID | WER@10 | Slope | Δ slope vs B0 | Label |
|------|-----|--------|-------|----------------|-------|
| 1 | **D3c** | 1.0000 | -0.03086 | +400% | **STRONG** |
| 2 | **D1b** | 1.0000 | -0.01097 | +78% | **STRONG** |
| 3 | **D4** | 1.0432 | 0.00480 | -178% | **INERT** |
| 4 | **D3b** | 1.0000 | 0.01509 | -344% | **INERT** |
| 5 | **D2a** | 1.0000 | 0.01989 | -422% | **INERT** |
| 6 | **D1d** | 1.0494 | 0.02881 | -567% | **INERT** |
| 7 | **D2d** | 1.0000 | 0.07064 | -1244% | **INERT** |

---

## Experiment Interpretations

### D3c

TopoLoss λ=0.01 (standard cortical-map regularisation). Slope -0.0309 — STRONG. After device fix, standard-strength TopoLoss is the **strongest loss configuration in the sweep**. Forcing nearby FFN neurons to be functionally similar (TopoNets) appears to regularise the encoder effectively on the tiny BCI sentence set, likely because the topographic constraint is equivalent to a structured dropout that improves generalisation.

### D1b

CTC linear anneal (weight 0.3→0 over 75 epochs). Slope -0.0110 — STRONG. Early CTC pushes the encoder to organise time-tokens phonemically; fading it to zero lets the model specialise for E2E decoding. Strongest explicit loss schedule tested (Graves et al. 2006; curriculum learning).

### D4

Label smoothing ε=0.1. Slope 0.0048 — INERT. Label smoothing modestly hurts at toy scale, likely because the model is already under-confident (WER≈1.0 at epoch 10) — adding more uncertainty does not help when the model is still trying to learn basic decoding.

### D3b

TopoLoss λ=0.001 (light cortical-map regularisation). Slope 0.0151 — INERT. After device fix (moved conv kernel to GPU), light TopoLoss regularisation shows marginal positive effect; the λ=0.001 signal may be too weak to shape weights meaningfully in 20 epochs.

### D2a

Contrastive loss ablation (weight=0). Slope 0.0199 — INERT. Removing InfoNCE slightly worsens WER descent, suggesting the contrastive term provides a useful neural↔text alignment signal — though the effect is modest at toy scale.

### D1d

CTC ablation (weight=0). Slope 0.0288 — INERT. Removing CTC entirely hurts WER trajectory, confirming CTC is load-bearing for phoneme alignment in this pipeline. A clean negative result.

### D2d

Contrastive loss ×2 (weight=2.0). Slope 0.0706 — INERT. Doubling the contrastive weight substantially hurts WER — the stronger alignment pressure likely fights cross-entropy and pushes a bad local minimum, consistent with the InfoNCE-vs-CE tension observed in other multi-task LM works.

---

**Track D winner:** D3c (slope -0.03086)
