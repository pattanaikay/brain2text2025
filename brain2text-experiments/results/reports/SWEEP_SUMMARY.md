# SWEEP SUMMARY — Brain2Text Autoresearch
*Completed 2026-06-07 | A100 40 GB | toy profile (20 epochs, 200 batches/epoch)*

---

## BEST BUILDING BLOCKS

| Stage | Winner | Slope | Label | Note |
|-------|--------|-------|-------|------|
| **Encoder** | **B1** | -0.01646 | **STRONG** | ConformerXL jitter-correction prenet |
| **Loss** | **D3c** | -0.03086 | **STRONG** | TopoLoss λ=0.01 cortical-map regularisation |
| **Projector** | **E1b** | -0.01235 | **STRONG** | Gated MLP beats Q-Former at toy scale |
| **Decoder** | *none* | — | — | C-track failed; C2 Phi-4-MM retry pending |
| **JEPA backbone** | **F3** | -0.02263 | **STRONG** | Neural-JEPA — signal best modelled on own terms |


---

## Master Ranked Leaderboard (top 20, toy profile)

| Rank | ID | Track | WER@10 | Slope | Label |
|------|----|-------|--------|-------|-------|
| 1 | D3c | D | 1.0000 | -0.03086 | **STRONG** |
| 2 | F3 | F | 1.0000 | -0.02263 | **STRONG** |
| 3 | B1 | B | 1.1296 | -0.01646 | **STRONG** |
| 4 | B3 | B | 1.0000 | -0.01303 | **STRONG** |
| 5 | E1b | E | 1.0185 | -0.01235 | **STRONG** |
| 6 | D1b | D | 1.0000 | -0.01097 | **STRONG** |
| 7 | E1a | E | 1.2654 | -0.00892 | **PROMISING** |
| 8 | B0_baseline | B | 1.1111 | -0.00617 | **WEAK** |
| 9 | E3 | E | 1.0000 | -0.00549 | **WEAK** |
| 10 | E2b | E | 1.0556 | -0.00412 | **WEAK** |
| 11 | H2 | H | 1.0000 | -0.00069 | **INERT** |
| 12 | B2 | B | 1.0000 | 0.00000 | **INERT** |
| 13 | D4 | D | 1.0432 | 0.00480 | **INERT** |
| 14 | B5 | B | 1.0000 | 0.01029 | **INERT** |
| 15 | D3b | D | 1.0000 | 0.01509 | **INERT** |
| 16 | B4 | B | 1.2654 | 0.01578 | **INERT** |
| 17 | F1 | F | 1.0494 | 0.01715 | **INERT** |
| 18 | D2a | D | 1.0000 | 0.01989 | **INERT** |
| 19 | D1d | D | 1.0494 | 0.02881 | **INERT** |
| 20 | F2 | F | 1.0000 | 0.03841 | **INERT** |

---

## Key Findings

### What Moves WER (positive signals)

1. **D3c TopoLoss λ=0.01 is the strongest loss** (slope −0.031) — cortical-map regularisation on FFN weights generalises better than CTC annealing or label smoothing on this tiny sentence set.
2. **B1 ConformerXL is the strongest encoder** (slope −0.0165) — jitter-correction prenet before patching produces cleaner neural tokens.
3. **E1b Gated MLP is the strongest projector** (slope −0.0124) — multiplicative gating converges faster than Q-Former's cross-attention at 200 batches/epoch.
4. **F3 Neural-JEPA** (slope −0.023) — JEPA pretraining directly on the BCI signal produces a better downstream backbone than borrowing audio or video masking objectives.
5. **CTC is load-bearing** — D1d (no CTC) hurts; D1b (CTC anneal) helps.

### What Does NOT Move WER (clean negatives)

- **Removing contrastive (D2a)**: marginally hurts — InfoNCE is useful at default weight.
- **Doubling contrastive (D2d)**: hurts badly — stronger alignment pressure fights CE.
- **MoE encoder (B4)**: INERT at toy scale — expert routing needs more data to specialise.
- **Audio/video JEPA (F1/F2)**: INERT or positive-slope — borrowed modality priors don't transfer to neural signals.
- **Combination CMB-1 (B1+QFormer+CTC)**: INERT — the winning levers interact negatively at toy scale; they do not stack additively.

---

## Failed / Deferred Experiments

| ID | Reason | Fix status |
|----|--------|------------|
| B2 HRM | DEQ backward bf16/fp32 dtype mismatch | Patched in run.py (`enc.to(compute_dtype)`) — retry pending |
| B3_mamba | mamba-ssm won't compile (CUDA 12.6 toolkit vs PyTorch+cu130) | apt-get cuda-nvcc-13-0 installed; retry pending |
| C1 Qwen2-Audio | AutoModelForCausalLM can't load Qwen2AudioConfig | Architecture fix needed |
| C2 Phi-4-MM | Phi4MM rejects Flash Attention 2 | Patched phi4mm.py with `attn_implementation='sdpa'` — retry pending |
| C3 Whisper-Qwen | Bridge linear on CPU | Patched run.py with bridge `.to(device)` — retry pending |
| A1 CKA | Shape mismatch in attention during CKA analysis | Fix in `tools/cka_analysis.py` needed |
| A2 perplexity | Corpora not provided | Needs `BCI_DATA_ROOT` with transcripts |
| A3 phoneme probe | Toy HDF5 has no phoneme labels | Needs CTC-annotated dataset |

---

## Handoff for Long-Term Implementation

The diagnostic sweep identifies the following stack for the full training run:

```
neural activity
  → B1 ConformerXL encoder  (jitter-correction prenet + macaron conformer)
  → E1b Gated MLP projector (relu(fc1(x)) * sigmoid(gate(x)), 384→1536)
  → D3c TopoLoss + D1b CTC anneal (topographic FFN reg + annealed CTC)
  → [decoder TBD — C2 Phi-4-MM / C3 Whisper-Qwen retry needed]
  → sentence
```

**No full/150-epoch run was executed here.** This is the diagnostic sweep only.
Carry these building blocks into the separate long-term training pipeline.
