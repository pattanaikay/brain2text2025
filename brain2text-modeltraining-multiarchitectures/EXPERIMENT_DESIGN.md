# Brain2Text — Comprehensive Experiment Design

**Last updated:** 2026-05-28  
**Target metric:** Word Error Rate (WER) < 0.10  
**Current baseline:** WER = 0.3673 (BIT + Qwen2.5-1.5B-Instruct, epoch 65)  
**Local hardware:** RTX 4050 Laptop GPU, 6 GB VRAM, CUDA 8.9, Python 3.11, PyTorch 2.5.1+cu121  
**Cloud hardware:** A100 40 GB (JarvisLabs instance)  

---

## How to Read This Document

Each experiment section specifies:
- **What** is being tested (the architectural or scientific question)
- **Why** it is expected to help (the mechanistic hypothesis)
- **How** to run it (code changes, commands, key flags)
- **Where** to run it (local 6 GB vs cloud A100)
- **What to measure** (the proxy metrics)
- **Decision rule** (when to promote a toy result to a full cloud run)

Sections are grouped into five tracks:

| Track | Question | Experiments |
|---|---|---|
| **A** | Does pretraining modality (audio vs image/video vs text) matter? | A1–A4 |
| **B** | Which encoder architecture extracts better neural representations? | B1–B5 |
| **C** | Which LLM decoder best bridges the neural–text gap? | C1–C3 |
| **D** | How should the loss functions be combined and scheduled? | D1–D4 |
| **E** | Does the projector architecture limit the modality bridge? | E1–E3 |

---

## Toy Training Protocol (Shared Across All Tracks)

Before any full A100 run, validate the hypothesis on a toy setup:

```
Full run:  150 epochs × 1369 batches × ~6 min/epoch = 15 hours  
Toy run:    20 epochs ×  200 batches × ~1 min/epoch = 20 minutes  
```

**Create the toy dataset once (run locally):**
```python
# scripts/create_toy_dataset.py
import h5py, random, shutil

def create_toy(full_path, toy_path, fraction=0.15, seed=42):
    random.seed(seed)
    with h5py.File(full_path, 'r') as f:
        keys = list(f.keys())
        kept = random.sample(keys, int(len(keys) * fraction))
        with h5py.File(toy_path, 'w') as g:
            for k in kept:
                f.copy(k, g)
    print(f"Toy: {len(kept)}/{len(keys)} samples")
```

**Toy training args:**
```bash
--epochs 20 --batch_size 4 --val_interval 2 --accumulation_steps 4 --lr 5e-5
```

**Ranking metric:** Do not compare final WER numbers between toy runs. Instead compare:
1. **WER at epoch 10** — measures how fast the model learns
2. **Slope** = (WER_epoch2 − WER_epoch20) / 18 — measures learning efficiency
3. **Best WER** achieved across all 20 epochs

A toy architecture beats the baseline if its slope is ≥ 3% better (relative).  
If slope improvement is 1–3%, combine it with the best other change and re-run toy.  
If improvement < 1%, discard.

---

## Track A — Pretraining Modality Study

### Scientific Question

Why does an audio-pretrained LLM (e.g. Aero1-Audio, Qwen2-Audio) outperform a
text-only LLM (Qwen2.5-1.5B) on neural decoding? Is it because:

- **H1 (Phoneme Prior):** Audio pretraining creates richer sub-word representations
  for phoneme sequences, which maps well to neural phoneme structure.
- **H2 (Spoken Language Prior):** Spoken text has different statistics than written text
  (shorter sentences, simpler syntax, repetition). Audio pretraining aligns the LLM
  with spoken sentence distributions.
- **H3 (Embedding Space Geometry):** The token embedding space of audio-pretrained LLMs
  is geometrically closer to what our BIT projector produces. The projector has a shorter
  distance to bridge.
- **H4 (Visual/Temporal Priors):** Pretraining on video (which is also a temporal
  sequential modality) might help similarly to audio, suggesting the benefit is about
  temporal structure, not speech specifically.

---

### A1 — Embedding Space CKA Analysis

**Question:** Which pretrained LLM backbone has an embedding space most aligned with our BIT encoder output?  
**Hypotheses tested:** H3  
**Can run locally:** ✅ Yes — no training, inference only  
**Expected runtime:** ~30 minutes on RTX 4050  

**Method:**
For each candidate LLM, run the validation set through both the frozen BIT encoder+projector
and the LLM's embedding layer. Compute linear CKA between the projected neural tokens
and the ground-truth text token embeddings.

```python
# tools/cka_analysis.py

import torch
import torch.nn.functional as F

def linear_cka(X, Y):
    """Centered Kernel Alignment (linear kernel). X, Y: (N, d) float tensors."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX = (X.T @ X)
    YtY = (Y.T @ Y)
    XtY = (X.T @ Y)
    return (XtY * XtY).sum() / ((XtX * XtX).sum() * (YtY * YtY).sum()).sqrt()

# Models to compare (load each, compute CKA, record):
CANDIDATES = [
    ("text-only-1.5B",    "Qwen/Qwen2.5-1.5B-Instruct"),
    ("text-only-7B",      "Qwen/Qwen2.5-7B-Instruct"),
    ("vision-3B",         "Qwen/Qwen2.5-VL-3B-Instruct"),   # image-pretrained backbone
    ("audio-7B",          "Qwen/Qwen2-Audio-7B-Instruct"),   # audio-pretrained backbone
    ("phi4-vision",       "microsoft/Phi-4-multimodal-instruct"),
]

# For each model:
# 1. Load model (no quantization to preserve embedding geometry)
# 2. Get embedding layer: model.get_input_embeddings()
# 3. For each val sample:
#    a. Run BIT_encoder + projector → neural_pooled: (384,) or (1536,) after projector
#    b. Tokenize ground truth sentence → embed → mean-pool → text_pooled: (llm_dim,)
# 4. Collect over val set → X: (N_val, projected_dim), Y: (N_val, llm_dim)
# 5. If dims differ, project X to llm_dim with a random linear map (just for CKA)
# 6. Compute CKA(X, Y) — higher = better alignment

# Expected result:
# audio-7B > text-only-7B > vision-3B ≈ text-only-1.5B > phi4-vision (uncertain)
# If audio ≈ text (both score high), H3 is false → look at H1/H2
```

**What the result tells you:**
- High CKA for audio → H3 is true → use Qwen2-Audio backbone
- High CKA for vision → temporal/sequential structure is the key, not speech-specific
- Low CKA for everything → projector bottleneck, not LLM choice

---

### A2 — Spoken Language Perplexity Test

**Question:** Is the LLM's language model better calibrated for spoken text vs written text?  
**Hypotheses tested:** H2  
**Can run locally:** ✅ Yes  
**Expected runtime:** ~15 minutes  

```python
# tools/perplexity_test.py

SPOKEN_CORPUS = [
    # The 250 BCI competition sentences (BCI utterances are spoken)
    # These are short, natural speech sentences
]

WRITTEN_CORPUS = [
    # Sample from Wikipedia / news articles (similar domain but written)
]

def compute_perplexity(model, tokenizer, sentences):
    losses = []
    for sent in sentences:
        ids = tokenizer(sent, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            loss = model(ids, labels=ids).loss
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))

# Compare:
# PPL_spoken vs PPL_written for each model
# Audio-pretrained model should have lower PPL_spoken / PPL_written ratio
# (more specialized for spoken text)
```

**Expected result:**
- All models: PPL_written < PPL_spoken (written text is more regular)
- Audio-pretrained model: smallest spoken/written PPL gap
- If all models have the same ratio → H2 is false → audio pretraining
  helps through a different mechanism (H1 or H3)

---

### A3 — Phoneme Probing Accuracy

**Question:** Does the LLM's hidden state at position t contain phoneme-level information?  
**Hypotheses tested:** H1  
**Can run locally:** ✅ Yes (small probe training)  
**Expected runtime:** ~45 minutes  

```python
# tools/phoneme_probe.py
# Train a linear probe on frozen LLM hidden states to predict phoneme labels.

# Data: BCI competition sentences with phoneme alignments
# (from the CTC training labels — phoneme_ids are already in your dataloader)

# For each LLM:
# 1. Embed ground-truth sentence → hidden states at last encoder layer
# 2. Train Linear(llm_dim, 42) with cross-entropy on phoneme labels
# 3. Measure phoneme classification accuracy (top-1 and top-5)

class PhonemeProbe(nn.Linear):
    pass  # Linear(llm_dim, 42)

# Audio-pretrained LLM should have significantly higher phoneme probe accuracy.
# If audio ≈ text on probe, H1 is false → mechanism is H2 or H3.
```

---

### A4 — Full E2E WER Comparison: Audio vs Image/Video Pretraining

**Question:** Does image/video pretraining help WER, and by how much vs audio?  
**Hypotheses tested:** H4 (temporal priors)  
**Can run locally:** ❌ Too large for 6 GB VRAM  
**Cloud:** A100 40 GB  
**Expected runtime:** 20 epochs × 3 min/epoch × 3 models = ~3 hours  

**Models to compare:**
```
Baseline:    Qwen/Qwen2.5-1.5B-Instruct     (text-only, small)
Audio-LLM:   Qwen/Qwen2-Audio-7B-Instruct    (audio-pretrained)  [architectures/audio_llm/qwen2_audio_e2e.py]
Vision-LLM:  Qwen/Qwen2.5-VL-3B-Instruct    (image/video pretrained, comparable size to audio-7B in 4-bit)
Split-Stack: WhisperQwen                     (frozen Whisper encoder + Qwen2.5-1.5B)  [architectures/audio_llm/whisper_qwen_e2e.py]
```

**Why Qwen2.5-VL for the vision comparison:**
Qwen2.5-VL is trained on images AND video (temporal sequences). It shares the same
Qwen backbone as our baseline. Using AutoModelForCausalLM on just the language backbone
gives us a LLM shaped by visual-temporal training, which is the cleanest comparison to
the audio-shaped backbone in Qwen2-Audio.

**Key architectural note for A4 — using audio/vision LLMs as text-only decoders:**
All audio_llm/ architectures bypass the native audio/vision encoder completely.
The BIT encoder's projected embeddings are injected as `inputs_embeds`. The LLM
backbone processes them as if they were token embeddings. The benefit is purely from
the weight initialization (shaped by audio/visual pretraining).

**Whisper-Qwen (split-stack) is the most principled audio experiment:**
```
BIT_Transformer → project to Whisper d_model (1280) → frozen Whisper encoder
                → bridge to Qwen2.5 d_model (1536) → Qwen2.5-1.5B decoder
```
This tests whether passing neural features through Whisper's learned speech manifold
(even when frozen) improves the LLM's ability to decode them. If yes, the Whisper
encoder is acting as a "speech normalizer" — mapping neural patterns into a space the
LLM was trained to read from.

**Running Whisper-Qwen locally:** Whisper-large-v3 (~2.9 GB) + Qwen2.5-1.5B-4bit (~1.2 GB)
= ~4.1 GB. **This fits on your 6 GB RTX 4050** for small batch sizes.

---

## Track B — Encoder Architecture Ablations

All encoder variants produce `(B, T_patch, 384)` output — they are drop-in
replacements for `BIT_Transformer` in `bit_e2e.py`. The E2E LLM decoder is
held constant at Qwen2.5-1.5B-Instruct with QLoRA.

---

### B1 — ConformerXL Encoder

**File:** `architectures/conformer/encoder.py`  
**Can run locally:** ✅ Yes  
**Memory footprint:** Similar to BIT — ConformerXL has 12 layers vs 7, but no session dict  

**What's different from BIT:**
1. **JitterCorrectionPrenet** runs BEFORE patching — three parallel dilated convolutions
   (dilation 1, 2, 4) fused by a 1×1 conv, followed by a BiGRU. This corrects neural
   spike-timing jitter before temporal downsampling.
2. **Macaron-style ConformerBlock** replaces TransformerBlock:
   - Half-weight FFN → Multi-head self-attention → Depthwise Conv module → Half-weight FFN
   - No RoPE — the conv module provides local temporal inductive bias instead
   - RMSNorm at block end instead of LayerNorm
3. **Universal read-in** — single `Linear(512, 512)` shared across all sessions.
   The paper argues ConvModule + BiGRU prenet provides enough local calibration.
4. **GroupNorm(8, dim)** inside ConvModule instead of BatchNorm — more stable at
   small batch sizes (batch_size=8-16).
5. 12 layers vs BIT's 7.

**Why it might help:**
Neural signals have spike-timing jitter: individual neurons fire within a ~10ms window
around the "true" firing time. The prenet's dilated conv + BiGRU explicitly learns to
smooth this jitter before the transformer sees the signal. Standard BIT treats each 20ms
bin independently in the patch embedding.

**Hypothesis:** JitterCorrectionPrenet reduces jitter → cleaner patch representations →
lower CTC PER → lower E2E WER.

**To run on toy data:**
```python
# In train_e2e.py, replace:
#   model.neural_encoder = BIT_Transformer(...)
# with:
from src.models.architectures.conformer.encoder import ConformerEncoder
model.neural_encoder = ConformerEncoder(
    input_dim=512, embed_dim=384, num_layers=12,
    patch_size=4, dropout=0.1, attn_dropout=0.1
)
# Note: SSL pretraining checkpoint won't load (different architecture).
# Start E2E from scratch OR load only the patch_embedding weights that match.
```

**Important: No SSL pretraining checkpoint for non-BIT encoders.**
All B1-B5 architectures must start E2E training from scratch (no pretrained encoder).
This is a significant disadvantage vs BIT. Compare fairly by also running a BIT
baseline WITHOUT loading the SSL/CTC checkpoint.

**Expected WER on toy:** May be higher than BIT at epoch 10 (cold start) but should
catch up by epoch 20 if the architecture is genuinely better.

---

### B2 — HRM (Hierarchical Reasoning Model) Encoder

**File:** `architectures/hrm/encoder.py`  
**Can run locally:** ✅ Yes (GRU-based, CPU-friendly for smoke tests)  
**Memory footprint:** Low — GRUCell-based, no large attention matrices  

**What's different:**
HRM processes neural data at two timescales simultaneously:

```
For each patch (100ms window):
  L-module (20ms): GRUCell iterates over 4 time bins
                   Input = x_t concat h_context_from_H
                   Output = h_l (local state)
  Fixed-point solver: iterate L-step until ||h - L(h)|| < 1e-3
  1-step gradient (custom autograd): O(1) memory instead of BPTT
  H-module (100ms): GRUCell over patch summaries
                    Input = h_l (last L-state)
                    Output = h_h (fed back as context to next patch's L-module)
```

**Why it might help:**
Neural speech decoding requires both fine-grained timing (individual spikes at 20ms)
and coarser word-level context (100ms windows). BIT's transformer treats all patches
equally. HRM's dual-timescale design mirrors the brain's own hierarchical processing
(primary motor cortex → premotor cortex → speech motor planning).

The DEQ fixed-point solver is the critical innovation: it allows L to "settle" on the
right hidden state for each 20ms bin before the H-module integrates across bins. This
is analogous to how each cortical layer reaches equilibrium before passing signals up.

**O(1) memory caveat:** The HRM uses `h_l = h_star.detach()` between patches — BPTT
does NOT flow across patch boundaries. Gradient only exists within a single patch's
fixed-point iterations. This is intentional for memory efficiency, but means the model
cannot learn long-range dependencies across patches via the recurrent path.

**To run:**
```python
from src.models.architectures.hrm.encoder import HRMEncoder
model.neural_encoder = HRMEncoder(patch_size=4, l_hidden=384)
```

**Pitfall:** fixed_point_solver iterates up to 10 times per patch. If T_patch=60,
that's 600 GRUCell forward passes per batch item. Profile first:
```python
# Smoke test timing before committing to toy run
import time
enc = HRMEncoder()
x = torch.randn(8, 240, 512)  # typical batch
t = time.time(); enc(x); print(f"HRM forward: {time.time()-t:.2f}s")
# BIT forward for reference
bit = BIT_Transformer(); t = time.time(); bit(x); print(f"BIT forward: {time.time()-t:.2f}s")
```

---

### B3 — MambaPOSSM (State Space Model) Encoder

**File:** `architectures/mamba_possm/encoder.py`  
**Can run locally:** ✅ Yes (GRU fallback if mamba-ssm unavailable)  
**GPU requirement for true Mamba:** CUDA 8.9+ ✅ (RTX 4050 is 8.9)  
**Install Mamba:** `pip install mamba-ssm causal-conv1d` (requires CUDA toolkit)  

**What's different:**

1. **IndividualSpikeTokenizer** replaces linear patch embedding:
   ```
   For each patch of 4 bins:
   64 learned queries cross-attend to 4 KV frames → sum-pool → 1 token
   ```
   Instead of `reshape + Linear(512*4, 384)`, cross-attention selects the most
   informative features from each spike frame. This should better handle the case
   where only 1-2 of the 4 bins in a patch contain strong spikes.

2. **SSMBlock** replaces TransformerBlock:
   - Mamba (selective SSM) processes sequences causally with O(L) memory vs O(L²) for attention
   - dt_min=0.01, dt_max=0.5: soft-window bias that regularizes the SSM's "selection" of
     relevant history
   - 3-layer input compression (0.4 dropout) before SSM blocks — aggressive regularization

3. **GRU fallback** when mamba-ssm unavailable — confirms architectural direction before
   spending time on Mamba install.

**Why it might help:**
Neural spike trains are fundamentally state-selective: at any given moment, only a
fraction of the 512 channels carry informative signals. The SSM's selective state space
(Mamba's S4 with input-dependent transitions) explicitly learns WHICH channels to track
over time — similar to how the brain's attention selects relevant neural populations.
Transformers assign attention to all pairs; Mamba learns a compact state representation.

**Cross-attention tokenizer is independently valuable:**
Even using the GRU fallback (no Mamba), the IndividualSpikeTokenizer replaces a hard
reshape with learned selection. Worth testing this component alone by combining it with
the BIT transformer backbone.

**To run (GRU mode first):**
```python
from src.models.architectures.mamba_possm.encoder import MambaPOSSMEncoder
model.neural_encoder = MambaPOSSMEncoder(
    patch_size=4, n_layers=7, ssm_backbone="gru", dropout=0.1, drop_path=0.1
)
```

**To run (Mamba mode, after installing mamba-ssm):**
```python
model.neural_encoder = MambaPOSSMEncoder(
    patch_size=4, n_layers=7, ssm_backbone="mamba", dropout=0.1, drop_path=0.1
)
```

**Sub-experiment B3a — Cross-attention tokenizer only:**
Replace only `BIT_Transformer`'s `patch_embedding` with `IndividualSpikeTokenizer`,
keep everything else (RoPE, 7 TransformerBlocks). This isolates the tokenizer's effect
from the SSM's effect.

---

### B4 — MoE (Mixture of Experts) Encoder

**File:** `architectures/moe/encoder.py`  
**Can run locally:** ✅ Yes (no special dependencies)  
**Extra loss term:** `model.neural_encoder.last_aux_loss` must be added to total loss  

**What's different:**
Replaces the TransformerBlock's MLP with SSMoEBlock:
- **6 specific experts** (routed, top-2 activated per token)
- **2 shared experts** (always active, average-weighted)
- TopK router with load-balance auxiliary loss (prevents expert collapse)

The architecture is identical to BIT in all other respects (same RoPE attention,
same patch embedding, same read-in). Only the FFN is replaced with MoE.

**Why it might help:**
Neural data is inherently multi-domain:
- Different neural populations encode phonemes, prosody, motor commands, etc.
- Different time patches correspond to different phases of speech production
- Different sessions have different signal quality and recording characteristics

A single FFN processes all of these the same way. MoE allows different tokens/patches
to be processed by specialized experts, which may learn domain-specific transformations.
The shared experts capture universal features; the routed experts specialize.

**To integrate the aux loss (required, else expert collapse within ~5 epochs):**
```python
# In train_e2e.py, after model.forward():
loss, ce_loss, contrastive_loss, ctc_loss = model(...)
if hasattr(model.neural_encoder, 'last_aux_loss'):
    moe_aux = model.neural_encoder.last_aux_loss * 0.01  # load-balance weight
    loss = loss + moe_aux
```

**What to monitor:**
- `moe_aux_loss` should be ~0.01-0.05 at convergence. If it spikes → expert collapse.
- Log expert utilization: count how often each of the 6 experts is in the top-2.
  All experts should be active ~uniformly.

**Sub-experiment B4a — How many experts?**
Try `n_specific=4` vs `6` vs `8`. Hypothesis: 6 matches well to the ~6 phoneme
feature categories (place, manner, voicing, nasality, stop/fricative, vowel height).

---

### B5 — ZenBrain Memory-Augmented Encoder

**File:** `architectures/zenbrain_memory/encoder.py`  
**Can run locally:** ✅ Yes (training mode, `use_memory=False`)  
**Memory mode (inference):** Requires buffer population from earlier trials  

**What's different:**
ZenBrainEncoder adds cross-attention over an episodic buffer to each transformer block:

```
Each DualMemoryBlock:
  1. Self-attention over current trial patches (same as BIT)
  2. Cross-attention: current patches (Q) × episodic buffer (K, V)   ← new
  3. FFN
```

The episodic buffer stores up to M=3000 high-confidence patch embeddings from previous
trials (CTC confidence > 0.7). During training, `use_memory=False` — cross-attention
is skipped entirely. At inference, the buffer is filled from earlier trials in the same
session.

**Why it might help:**
The BCI system's neural signals vary day-to-day and trial-to-trial (electrode drift,
noise, fatigue). By caching high-confidence representations from earlier in the same
session, ZenBrain can use clean examples to guide decoding of noisier trials — similar
to in-context learning but for neural patterns.

**The "zero-calibration" angle:**
Traditional BCI systems require explicit recalibration each day. ZenBrain's buffer
fills automatically from high-confidence trials (no labels needed), making it
zero-calibration at inference time.

**Training procedure:**
1. Train normally with `use_memory=False` (identical to BIT)
2. At inference, enable `use_memory=True` and populate buffer:
   ```python
   model.neural_encoder.use_memory = True
   # After each batch:
   model.neural_encoder.update_buffer_from_outputs(neural_tokens, ctc_logits)
   ```

**Experiment design:**
Run val set twice:
- Pass 1 (cold buffer): `use_memory=False` — baseline WER
- Pass 2 (warm buffer, populated from pass 1): `use_memory=True` — memory-boosted WER
Difference in WER = pure benefit of episodic memory.

**Local viability:** Buffer is (3000, 384) float32 = 4.6 MB. Trivial.

---

## Track C — LLM Decoder Variants (Audio Pretraining)

All C experiments use the BIT encoder baseline. Only the LLM decoder changes.
See also Track A for the scientific motivation.

---

### C1 — Qwen2-Audio-7B-Instruct Decoder

**File:** `architectures/audio_llm/qwen2_audio_e2e.py`  
**Can run locally:** ❌ 7B model at 4-bit ≈ 7-9 GB, exceeds 6 GB VRAM  
**Cloud:** A100 40 GB required  
**LLM dim:** 3584  
**Prompt tokens:** Uses `<|audio_bos|>` / `<|audio_eos|>` special tokens from Qwen2-Audio  

**Key implementation detail:**
`AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")` loads the
full Qwen2-Audio model but the class is `Qwen2AudioForConditionalGeneration`, NOT a
standard causal LM. The provided implementation works around this by using
`AutoModelForCausalLM` which falls back to the language backbone.

**Verify before starting full run:**
```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct",
                                           trust_remote_code=True, device_map="cpu")
print(type(m))  # Should NOT be Qwen2AudioForConditionalGeneration
print(m.config.hidden_size)  # Should be 3584
```

**Expected WER:** 0.25–0.30 on full training (vs baseline 0.3673)

---

### C2 — Phi-4-Multimodal-Instruct Decoder

**File:** `architectures/audio_llm/phi4_mm_e2e.py`  
**Can run locally:** ❌ 5.6B at 4-bit ≈ 6-7 GB, borderline  
**LLM dim:** 3072  
**LoRA targets:** `["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]` — note fused QKV  

**Why Phi-4-multimodal:**
Phi-4-multimodal was pretrained on speech (Azure Speech data), video, and images.
Microsoft describes it as having "speech-to-text understanding" at 5.6B parameters —
smaller than Qwen2-Audio-7B but with multimodal pretraining that includes both
temporal (video) and acoustic (speech) signals.

**Local attempt (with batch_size=1, gradient checkpointing):**
```bash
py scripts/train_e2e_local.py \
  --model phi4_mm \
  --no_quantize False \
  --batch_size 1 \
  --accumulation_steps 16 \
  --epochs 5  # just to check if it fits
```

**Fused QKV note:** Phi-4 uses `qkv_proj` (one fused matrix) instead of separate
`q_proj`, `k_proj`, `v_proj`. LoRA applied to `qkv_proj` covers all three heads.
Make sure the LoRA target list in the config is correct.

---

### C3 — Whisper-Qwen Split Stack

**File:** `architectures/audio_llm/whisper_qwen_e2e.py`  
**Can run locally:** ✅ Yes — Whisper-large-v3 (2.9 GB) + Qwen2.5-1.5B-4bit (1.2 GB) ≈ 4.1 GB  
**This is the most important C experiment for local testing**  

**Architecture:**
```
BIT_Transformer (384) → Linear(384, 1280) + LN → Whisper Encoder (frozen) (1280)
                      → Linear(1280, 1536) → Qwen2.5-1.5B-Instruct + LoRA (1536)
```

**Why this is scientifically interesting:**
The Whisper encoder has learned a "speech manifold" — a representation space where
acoustically similar speech signals cluster together. Projecting neural features into
this space and then passing through the frozen Whisper encoder asks: does the Whisper
encoder provide useful structure even for signals it was never trained on?

If yes: the speech manifold is general enough to organize neural speech representations.
This supports H3 (embedding alignment hypothesis).

If no: the Whisper encoder blocks information rather than helping — neural signals don't
live in the acoustic manifold even after linear projection.

**Whisper is frozen — only these modules are trained:**
- `BIT_Transformer` (loaded from CTC checkpoint)
- `neural_to_whisper` (Linear 384→1280 + LN) — new, randomly initialized
- `whisper_to_llm` (Linear 1280→1536) — new, randomly initialized
- Qwen2.5-1.5B with LoRA

**To run locally:**
```bash
py scripts/train_e2e_local.py \
  --encoder whisper_qwen \
  --train_h5 data/toy_train.h5 \
  --val_h5 data/val.h5 \
  --batch_size 4 \
  --epochs 20
```

---

## Track D — Loss Function Design

The current loss is:
```
total = CE_loss + 0.3 × CTC_loss + 1.0 × Contrastive_loss
```
All weights are fixed across all 150 epochs.

---

### D1 — CTC Weight Annealing

**Question:** Should CTC loss be strong early (force phoneme learning) and weak late (let CE dominate)?  
**Can run locally:** ✅ Yes — no architecture changes  
**Code change:** Single LR schedule line  

**Rationale:**
CTC loss trains the encoder to produce phoneme-decodable representations. This is most
valuable early in training when the encoder is random. By epoch 75+, the encoder already
encodes phonemes (PER=0.40 from the CTC pretraining stage). At this point, the CTC loss
is a constraint that may prevent the encoder from specializing further for E2E decoding.

**Annealing schedule:**
```python
# In train_e2e.py, replace ctc_weight=0.3 with:
ctc_weight = max(0.0, 0.3 * (1.0 - epoch / 75))
# Epoch 0:  ctc_weight = 0.30
# Epoch 37: ctc_weight = 0.15
# Epoch 75: ctc_weight = 0.00 (CTC fully off)
```

**Variants to compare:**
- D1a: Fixed 0.3 (current baseline)
- D1b: Linear anneal 0.3→0.0 over 75 epochs
- D1c: Step anneal: 0.3 until epoch 50, then 0.1, then 0.0 after epoch 100
- D1d: CTC=0.0 from the start (full ablation — does CTC help at all in E2E?)

---

### D2 — Contrastive Loss Temperature and Weight

**Question:** Is the learnable temperature in `ModalityAlignmentLoss` well-calibrated?  
**Can run locally:** ✅ Yes  

```python
# Current: temperature = nn.Parameter(torch.tensor(0.07)) — learnable
# But: this is with contrastive_weight = 1.0 (implicitly)
# Consider: separate the weight from the temperature:
contrastive_loss = contrastive_weight × contrastive_loss_fn(neural_pooled, text_pooled)
```

**Variants:**
- D2a: contrastive_weight=0.0 (full ablation)
- D2b: contrastive_weight=0.1
- D2c: contrastive_weight=1.0 (current)
- D2d: contrastive_weight=2.0

**Important diagnostic:**
If contrastive_weight=0.0 gives BETTER WER than 1.0, the contrastive loss is actually
hurting by conflicting with CE. This would suggest the neural pooled embeddings and
text pooled embeddings are too different to align with a single projection, and the
contrastive loss is pushing the model into a bad local minimum.

---

### D3 — Topographic Loss (TopoLoss)

**File:** `architectures/topoloss/topo_loss.py`, `architectures/topoloss/hooks.py`  
**Can run locally:** ✅ Yes — pure regularizer, no new parameters  
**Adds no trainable parameters; only a loss term**  

**What it does:**
Forces FFN weight matrices to become spatially organized like cortical maps — neurons
that are "nearby" in the FFN's hidden dimension become functionally similar (measured
by cosine similarity after Gaussian blur).

```
W: (d_hidden, d_in) → reshape to (d_in, sqrt(d_hidden), sqrt(d_hidden))
Apply Gaussian blur to the 2D neuron grid
Loss = -mean cosine similarity between W and blurred_W
```

**Rationale from TopoNets:**
Models with topographically organized layers generalize better and are more robust to
input perturbation — because nearby neurons encode related features, novel inputs can
be decoded by the local neighborhood even if individual neurons haven't seen that
exact pattern.

**Implementation:**
```python
from src.models.architectures.topoloss.topo_loss import TopoLoss
from src.models.architectures.topoloss.hooks import collect_ffn_first_linears

# After model creation:
topo_targets = collect_ffn_first_linears(model.neural_encoder)
topo_loss_fn = TopoLoss(topo_targets, sigma=1.0)

# In training loop:
topo_loss = topo_loss_fn() * topo_weight  # topo_weight = 0.01
total_loss = ce_loss + ctc_weight * ctc_loss + contrastive_loss + topo_loss
```

**Note:** `d_hidden=1024` in BIT's FFN = sqrt(1024) = 32. Perfect square ✅
`d_hidden=384` (embed_dim) in ConformerEncoder = sqrt(384) ≈ 19.6. NOT a perfect square.
TopoLoss pads to next perfect square (400 = 20²). No issue, but slight inefficiency.

**Variants:**
- D3a: topo_weight=0.0 (baseline)
- D3b: topo_weight=0.001
- D3c: topo_weight=0.01
- D3d: topo_weight=0.1 (likely too strong — watch for gradient conflict)

---

### D4 — Label Smoothing in CE Loss

**Can run locally:** ✅ Yes  
**One-line change**  

```python
# Current: standard cross-entropy via LLM's built-in loss
# Proposed: pass label_smoothing to the loss computation

# Option 1: Use a custom CE with smoothing
loss = F.cross_entropy(shift_logits.reshape(-1, vocab_size),
                       shift_labels.reshape(-1),
                       ignore_index=-100,
                       label_smoothing=0.1)

# Option 2: Stick with HF's built-in loss but wrap with smoothing
```

**Rationale:** With small BCI datasets (~1000 samples), the model can overfit to exact
token sequences. Label smoothing with ε=0.1 prevents the model from becoming overconfident
on training sentences.

---

## Track E — Projector Architecture

The projector is the most underexplored component. It performs a fundamental dimensionality
expansion (384 → 1536) and semantic shift (neural patterns → text token space).

---

### E1 — Deeper / Wider MLP Projector

**Can run locally:** ✅ Yes  

```python
# Current: 3-layer MLP with hidden_dim=1024
# Proposed variants:

# E1a: 5-layer MLP
class DeepMLPProjector(nn.Module):
    def __init__(self, input_dim=384, output_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),      nn.ReLU(),
            nn.Linear(1024, 2048),      nn.ReLU(),
            nn.Linear(2048, 1024),      nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.LayerNorm(output_dim)
        )

# E1b: Gated MLP (adds multiplicative interaction)
class GatedMLPProjector(nn.Module):
    def __init__(self, input_dim=384, output_dim=1536):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.gate = nn.Linear(input_dim, 1024)  # sigmoid gate
        self.fc2 = nn.Linear(1024, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x)) * torch.sigmoid(self.gate(x))
        return self.ln(self.fc2(h))
```

---

### E2 — Q-Former Cross-Attention Projector

**Can run locally:** ✅ Yes (lightweight cross-attention)  
**Impact estimate:** High — this is the architectural change most likely to give a
meaningful WER improvement on the projector side.  

**Rationale:**
The current MLP projector processes each neural token independently — token at position t
produces output at position t with no interaction with tokens at other positions. A
Q-Former allows the output representations to be informed by the FULL neural sequence.

```python
class QFormerProjector(nn.Module):
    """
    N_QUERIES learned tokens attend to the full neural sequence via cross-attention,
    then are projected to LLM embedding dim. Reduces sequence length from T_patch
    to N_QUERIES (e.g., 32), which is a constant regardless of neural sequence length.
    """
    def __init__(self, input_dim=384, output_dim=1536, n_queries=32, n_heads=6):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_queries, input_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(input_dim, n_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(input_dim)
        self.ln_kv = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.ln_out = nn.LayerNorm(output_dim)

    def forward(self, neural_tokens, key_padding_mask=None):
        # neural_tokens: (B, T_patch, 384)
        B = neural_tokens.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # (B, N_Q, 384)
        kv = self.ln_kv(neural_tokens)
        out, _ = self.cross_attn(self.ln_q(q), kv, kv,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False)
        return self.ln_out(self.proj(out))  # (B, N_Q, 1536)
```

**Key effect:** The output sequence is always N_QUERIES=32 tokens long, regardless of
neural input length. This means:
- No padding waste in attention (all 32 LLM tokens carry signal)
- Better batch efficiency (sequences are all the same length)
- The queries can learn to focus on phoneme-salient time windows

**Variants:**
- E2a: N_QUERIES=16 (very compressed)
- E2b: N_QUERIES=32 (recommended starting point)
- E2c: N_QUERIES=64 (closer to current variable length)

---

### E3 — Patch Size vs Projector Interaction

**Can run locally:** ✅ Yes  

The current setup: patch_size=4 (80ms windows) vs paper's patch_size=5 (100ms).
A larger patch_size means fewer but richer tokens. Combined with the Q-Former:

```
patch_size=4 → ~60 neural tokens → Q-Former(32 queries) → 32 LLM tokens
patch_size=8 → ~30 neural tokens → Q-Former(32 queries) → 32 LLM tokens (more compressed at source)
patch_size=5 → ~48 tokens       → Q-Former(32 queries) → 32 LLM tokens (paper setting)
```

Run a 2×3 grid search on toy data:
| patch_size | N_QUERIES | expected WER slope |
|---|---|---|
| 4 | 32 | baseline |
| 5 | 32 | likely better (paper-matched) |
| 8 | 32 | unknown |
| 4 | 16 | more aggressive compression |
| 5 | 16 | |
| 5 | 64 | less compression |

---

## Local Hardware Feasibility Matrix

| Experiment | Model (frozen + trainable) | Est. VRAM | Runs Locally? |
|---|---|---|---|
| **A1** CKA Analysis | BIT + 5 LLMs (inference only) | 4-8 GB per model | ✅ One at a time |
| **A2** Perplexity | 5 LLMs (inference) | 2-8 GB per model | ✅ |
| **A3** Phoneme Probe | LLM + linear probe | 4-8 GB | ✅ |
| **B1** ConformerXL + Qwen1.5B | ~3 GB | ✅ (batch_size=2) |
| **B2** HRM + Qwen1.5B | ~2.5 GB | ✅ |
| **B3** MambaPOSSM + Qwen1.5B | ~3 GB | ✅ (needs mamba-ssm) |
| **B4** MoE + Qwen1.5B | ~3.5 GB | ✅ |
| **B5** ZenBrain + Qwen1.5B | ~3 GB | ✅ |
| **C1** Qwen2-Audio-7B | ~8 GB (4-bit) | ❌ OOM |
| **C2** Phi4-MM | ~7 GB (4-bit) | ❌ borderline OOM |
| **C3** WhisperQwen | ~4.1 GB | ✅ |
| **D1-D4** Loss variants | Same as BIT baseline | ~3 GB | ✅ |
| **E1** Deep MLP Projector | +negligible | ~3 GB | ✅ |
| **E2** Q-Former Projector | +2 MB | ~3 GB | ✅ |
| **E3** Patch size grid | Negligible | ~3 GB | ✅ |

**Local batch size guidance (6 GB VRAM):**
```
batch_size=2, accumulation_steps=8  → effective batch_size=16
gradient_checkpointing=True          → saves ~30% VRAM at 20% speed cost
no_quantize=False                    → use 4-bit NF4 for LLM
```

---

## Priority Order for Running Experiments

### Phase 1 — Cheap Analysis (Local, No Training, This Week)
Run A1 (CKA), A2 (perplexity), A3 (phoneme probe) on local machine.  
These answer the scientific question about audio pretraining without any GPU training.  
Expected time: 2 hours total.

### Phase 2 — Toy Encoder Ablations (Local, 20-Epoch Toy Runs)
Run B1, B2, B3 (GRU mode), B4, B5 on toy dataset.  
Each run: ~20 minutes on RTX 4050 with batch_size=2.  
**Baseline to compare against:** BIT + Qwen1.5B from scratch (no SSL checkpoint),
same 20 epochs, same toy data.  
Expected time: ~3 hours for 6 toy runs.

### Phase 3 — Toy Loss and Projector Ablations (Local)
Run D1b, D2a/d, D3b/c, E2b (Q-Former), E3 patch size grid.  
Each run: ~20 minutes. Total: ~5 hours.

### Phase 4 — Winners to Full Cloud Training (A100)
Take the top 2 encoder architectures from Phase 2 and best loss/projector combo from
Phase 3. Run each for 80 epochs on full data.  
Expected time: ~3 hours per experiment.

### Phase 5 — Audio LLM Experiments (A100 Only)
Run C1 (Qwen2-Audio) and A4 (vision LLM comparison).  
Run C3 (WhisperQwen) locally first as a proxy.

---

## Decision Rules for Promoting Toy Results

```
Toy WER slope improvement vs baseline:
  ≥ 5% relative  → Strong signal: run full 80-epoch cloud experiment immediately
  3-5% relative  → Promising: combine with best other change, re-run toy
  1-3% relative  → Weak signal: only pursue if compute is free
  < 1% relative  → Discard

Val loss divergence on toy:
  If val loss rises faster than baseline → overfitting; add dropout before promoting
  If val loss flat but WER improves → calibration issue; acceptable, promote anyway
```

---

## Diagnostic Checklist Before Each Experiment

Before running any toy experiment, verify:
1. `[DBG] n_valid_tokens > 0` (labels are not all masked)
2. `ce_loss_batch < 10.0` at epoch 0 (LLM is not completely off)
3. `ctc_loss < 5.0` at epoch 0 (CTC head initializes reasonably)
4. GPU memory usage: `nvidia-smi` shows headroom before OOM
5. For MoE: `moe_aux_loss ≈ 0.01-0.05` (experts not collapsed)
6. For ZenBrain memory mode: buffer is non-empty before pass 2

---

## Reference: Architecture Summary Table

| Architecture | Encoder Type | Key Innovation | New Params vs BIT | Session-Specific? |
|---|---|---|---|---|
| **BIT** (baseline) | Transformer + RoPE | MAE SSL pretraining | — | ✅ ModuleDict read-in |
| **ConformerXL** | Conformer + JitterPrenet | Dilated conv + BiGRU jitter correction before patching | +~3M (prenet) | ❌ Universal |
| **HRM** | Dual-timescale GRU | Fixed-point DEQ, 20ms + 100ms timescales | −10M (no attention) | ❌ |
| **MambaPOSSM** | SSM + cross-attn tokenizer | Selective state space + attention-based tokenization | +~1M (tokenizer) | ❌ |
| **MoE** | Transformer + RoPE + MoE FFN | 6-expert routing replaces FFN, load-balance loss | +~40M (experts) | ❌ |
| **ZenBrain** | Transformer + RoPE + episodic buffer | Cross-attention over cached high-conf patches | +~2M (cross-attn) | ❌ |
| **Qwen2-Audio decoder** | BIT (unchanged) | Audio-pretrained 7B LLM backbone | +~9B (vs 1.5B) | ✅ |
| **Phi4-MM decoder** | BIT (unchanged) | Speech+video pretrained 5.6B backbone | +~7B | ✅ |
| **WhisperQwen** | BIT + frozen Whisper encoder | Speech manifold bridge between neural and text LLM | +615M (Whisper) | ✅ |
| **+ TopoLoss** | Any encoder | Cortical topographic regularization on FFN weights | 0 (loss only) | N/A |

---

*This document covers all architectures present in `src/models/architectures/`. New
architectures added to the folder should follow the same drop-in interface:
`forward(x, session_id=None, mask_patches=None, neural_lengths=None)` returning
`(B, T_patch, 384)`.*
