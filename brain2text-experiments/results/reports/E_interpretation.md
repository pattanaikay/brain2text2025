# Track E — Projector Architecture
*Generated 2026-06-07 — toy-profile sweep, 20 epochs, 200 batches/epoch*

**Question:** Is the projector the underexplored bottleneck in the neural→text semantic bridge?

**Baseline for comparison:** B0_baseline (BIT from scratch), slope = -0.00617

---

## Ranked Results

| Rank | ID | WER@10 | Slope | Δ slope vs B0 | Label |
|------|-----|--------|-------|----------------|-------|
| 1 | **E1b** | 1.0185 | -0.01235 | +100% | **STRONG** |
| 2 | **E1a** | 1.2654 | -0.00892 | +44% | **PROMISING** |
| 3 | **E3** | 1.0000 | -0.00549 | -11% | **WEAK** |
| 4 | **E2b** | 1.0556 | -0.00412 | -33% | **WEAK** |

---

## Experiment Interpretations

### E1b

Gated MLP projector (relu(fc1(x))·σ(gate(x))). Slope -0.0123 — STRONG. Multiplicative gating (Dauphin et al. 2017; Shazeer 2020) provides input-dependent feature selection without adding many parameters, enabling faster convergence than either deeper MLP (E1a) or cross-attention Q-Former (E2b) at toy scale.

### E1a

Deep MLP projector (5-layer, bottleneck 2048). Slope -0.0089 — PROMISING. More depth adds representational power but also more parameters to optimise; the improvement over the 3-layer baseline is modest at 200 batches/epoch.

### E3

Patch×Query grid search (patch_size∈{4,5,8} × n_queries∈{16,32,64}). Slope -0.0055 — WEAK. The grid shows no configuration outperforms the default patch=4, queries=32 at toy scale; aggressive patch compression (ps=8) loses temporal resolution faster than the Q-Former can recover it.

### E2b

Q-Former projector (32 queries, 2 cross-attention layers). Slope -0.0041 — WEAK. Cross-attention over the full neural sequence with 32 learned queries (BLIP-2 style; Li et al. 2023) underperforms gated MLP at this data scale. The queries need many steps to learn which temporal windows are phoneme-salient — a data-hungry component.

---

**Track E winner:** E1b (slope -0.01235)
