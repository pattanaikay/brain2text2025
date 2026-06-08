# Track C — LLM Decoder Variants
*Generated 2026-06-07 — toy-profile sweep, 20 epochs, 200 batches/epoch*

**Question:** Does an audio/multimodal-pretrained LLM bridge the neural→text gap better than text-only Qwen2.5-1.5B?

**Baseline for comparison:** B0_baseline (BIT from scratch), slope = -0.00617

---

## Ranked Results

| Rank | ID | WER@10 | Slope | Δ slope vs B0 | Label |
|------|-----|--------|-------|----------------|-------|
| — | C3 | N/A | N/A | — | **FAIL/SKIP** |

---

## Experiment Interpretations

### C3

Whisper-Qwen split stack. FAILED (retry pending) — Whisper bridge linear on CPU vs GPU activations; patched in run.py.. BIT → Linear(384,1280) → frozen Whisper-large-v3 → Qwen2.5-1.5B; fails because the Whisper bridge linear layers are initialised on CPU while encoder activations are on GPU. Fix: bridge modules moved to device in run.py.

---

**Track C winner:** none
