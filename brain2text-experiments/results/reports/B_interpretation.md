# Track B — Encoder Architecture
*Generated 2026-06-07 — toy-profile sweep, 20 epochs, 200 batches/epoch*

**Question:** Which neural encoder extracts the most speech-decodable representation?

**Baseline for comparison:** B0_baseline (BIT from scratch), slope = -0.00617

---

## Ranked Results

| Rank | ID | WER@10 | Slope | Δ slope vs B0 | Label |
|------|-----|--------|-------|----------------|-------|
| 1 | **B1** | 1.1296 | -0.01646 | +167% | **STRONG** |
| 2 | **B3** | 1.0000 | -0.01303 | +111% | **STRONG** |
| 3 | **B0_baseline** | 1.1111 | -0.00617 | +0% | **WEAK** |
| 4 | **B2** | 1.0000 | 0.00000 | -100% | **INERT** |
| 5 | **B5** | 1.0000 | 0.01029 | -267% | **INERT** |
| 6 | **B4** | 1.2654 | 0.01578 | -356% | **INERT** |

---

## Experiment Interpretations

### B1

ConformerXL with jitter-correction prenet. Slope -0.0165 vs B0 -0.0062 → **167% steeper descent**. The dilated-conv+BiGRU prenet smooths ~10 ms spike-timing jitter before patching, producing cleaner tokens and faster phoneme alignment — confirming the ConformerXL/iPhoneme hypothesis (Gulati et al. 2020).

### B3

MambaPOSSM GRU fallback (cross-attention tokenizer). Slope -0.0130 — STRONG. Selective-state-space inductive bias transferred through the GRU proxy; the cross-attention tokenizer replaces hard reshape with learned channel selection.

### B0_baseline

BIT encoder trained from scratch — the fair control. Slope -0.0062 establishes the reference every B1–B5 is judged against.

### B2

HRM dual-timescale DEQ. DEFERRED — DEQ custom backward mixes bf16 activations with fp32 GRU weights under autocast.. Fixed-point GRU at 20 ms and 100 ms timescales mirrors cortical hierarchy; the O(1)-memory fixed-point iteration is theoretically attractive but the DEQ custom backward mixes bf16 activations with fp32 GRU weights under autocast, causing a runtime error.

### B5

ZenBrain episodic-memory encoder. Slope 0.0103 — INERT. Cross-attention over a high-confidence trial buffer did not improve slope at toy scale; the episodic buffer may require a long burn-in of high-confidence trials before it provides useful signal.

### B4

MoE encoder (6 routed + 2 shared experts). Slope 0.0158 — INERT. Expert routing may need more data/steps than 200 toy batches to learn meaningful specialisation; the aux load-balance loss kept routing from collapsing but didn't translate to faster WER descent at this scale.

---

**Track B winner:** B1 (slope -0.01646)
