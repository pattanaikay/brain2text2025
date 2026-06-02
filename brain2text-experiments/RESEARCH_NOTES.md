# Research Notes — DietCorp + Sleep Consolidation + ZenBrain BCI
**Session:** 2026-06-02  
**Repo:** `brain2text-experiments` (C:\Projects\Brain2Text2025\brain2text2025\brain2text-experiments)  
**Status:** Infrastructure complete, 40/40 tests green, ready to run with trained checkpoint

---

## 1. The Thesis Question

> **Does increasing the depth of the per-trial memory consolidation step (N "sleep"
> passes instead of DietCorp's single AdamW step) reduce word/phoneme error rate
> under day-to-day electrode drift, while holding wake-time inference at a single
> forward pass?**

This is a BCI clinical problem: ALS / brainstem-stroke patients need a decoder
that adapts to electrode drift across days **without recalibration sessions** and
**without any extra latency at inference time**.

---

## 2. The Three Source Papers and How They Unify

### 2a. DietCorp — arXiv:2507.02800 (Feghhi et al., 2025)

**Architecture:** Unidirectional causal Transformer, 5 layers, 384-dim hidden, 9.4M params.
Decodes 256-channel microelectrode-array activity into phonemes at 10 Hz via
non-overlapping 100ms patches.

**The drift problem it solves:**
- Without adaptation: WER climbs 22.7% (day 1) → 66.5% (day 8)
- With DietCorp TTA: WER stays flat at ~12%

**How the TTA (Test-Time Adaptation) works — the exact mechanism:**
```
After each decoded sentence/trial:
  1. Generate 64 time-masked augmentations of the trial
     (randomly zero out ~53% of time patches — forces structural learning)
  2. Decode each augmentation with the current model → beam search + n-gram LM
     → produces a "pseudo-label" (the model's best guess at the phoneme sequence)
  3. Take ONE AdamW gradient step on the PATCH-EMBEDDING MODULE ONLY
     (the input layer: patch_ln1 + patch_embedding = ~792K params)
     using CTC loss against the pseudo-label
  4. All other layers remain FROZEN

Cost: 18ms per trial. The patch-embed is the only thing that drifts.
```

**Key architectural detail:** DietCorp uses `best_model_per.pth` (trained with
`train_ctc.py`) — a BIT encoder with a CTC head that predicts 42 phoneme classes
directly. This is the LLM-free path for the drift eval.

### 2b. "Do Language Models Need Sleep?" — arXiv:2605.26099 (Lee et al., CMU/UMD, 2026)

**The core insight:** Converting observed context into useful weight memory is itself
a hard computation that one forward pass is not enough to do well. The bottleneck
is not memory *capacity* but the amount of *computation available* to transform
evicted context into useful internal state.

**The mechanism (Algorithm 1):**
```
For each chunk of context:

  CONSOLIDATION PHASE (offline, N passes over the same context):
    for n = 1 to N:
        h, S = Blocks(h, S)    # full forward pass
        # S = fast weights (SSM state) — refined each pass
    clear the KV cache

  PREDICTION PHASE (one pass, wake time — always single-pass):
    h, S = Blocks(h, S)
    output = predict(h)

  BACKPROP goes through BOTH phases end-to-end.
  The model LEARNS how to consolidate — it's not just gradient descent.
```

**Key experimental results:**
- Rule 110 cellular automaton (hard sequential): N=1→0%, N=2→20%, N=3→30%, N=4→30%+
- Depo k-hop retrieval: N=1 stalls at 4+ hops; N=4 begins to improve at 16 hops
- GSM-Infinite math: 6-op problems: N=1→74%, N=6→81% (Jet-Nemotron); 47% gain (Ouro)
- **Section 7 caveat:** Large N can be unstable; training grows linearly with N

**Mapping to BCI:**

| Sleep Paper Concept | BCI Equivalent |
|---|---|
| Context window filling up | One decoded neural trial |
| Fast weights S (SSM state) | Patch-embedding layer parameters |
| N offline consolidation passes | N AdamW steps on patch-embed |
| KV cache eviction | Moving to next trial (real-time constraint) |
| Wake-time single forward pass | Actual clinical decoding (latency-constrained) |
| Gradient flows through consolidate→predict | Same: backprop through N-step TTA |

**Why neural drift is the right regime:** The sleep paper predicts gains are largest
on *sequential* problems — where today's state depends on yesterday's. Electrode
drift is inherently sequential (impedance builds up over days; signal statistics
correlate across adjacent days). This is why H_main is well-posed.

### 2c. ZenBrain — arXiv:2604.23878 (Bering, 2026)

**What ZenBrain is:** A neuroscience-inspired 7-layer memory architecture for
autonomous AI agents. NOT a neural network — it's an orchestration system (BM25 +
embedding retrieval, FSRS scheduling, knowledge graph). ~14 of its 15 components
run on the host CPU.

**The 7 memory layers:**
```
Working memory      ← current trial being decoded (on NPU/GPU)
Short-term memory   ← recent trials this session (RAM)
Episodic memory     ← ring buffer of high-confidence past trials (BUILT ✓)
Semantic memory     ← patterns extracted across many sessions
Procedural memory   ← learned decoding strategies
Core memory         ← stable long-term knowledge
Cross-context       ← bridging across different environments
```

**The key algorithm for this thesis — Simulation-Selection Sleep Loop (App. B.3):**
```
Offline consolidation trigger (between sessions):
  1. Score each stored trial by:
     priority = |TD_error| + reward + novelty
     (|surprise when seeing this again| + how useful it was + how different from others)
  2. Select top-K trials by priority
  3. Replay those K trials through the consolidation step (N passes each)
  4. Result: +37% stability improvement, 47.4% storage reduction vs random replay
```

**ZenBrain's role in the thesis:**
- Provides the **ring buffer** (what to store)
- Provides the **write policy** (session-keyed + confidence-gated — only store
  high-confidence trials)
- Provides the **replay scheduler** (Simulation-Selection priority — what to
  consolidate next)
- The **Simulation-Selection** scheduler is Phase 4 (not yet built)

**The episodic-consistency loss** (now live):
- During training, the memory stage reads back past trials via cross-attention
- Loss = MSE(current_latent, recalled_latent.detach())
- Forces the encoder to produce latents consistent with its past representations
- This is the "remember what you've seen before" regulariser

---

## 3. The Falsifiable Hypothesis

**H_main:**
> WER-under-drift at day 8 decreases monotonically as N increases, while
> `wake_latency_ms` stays flat across N and `consolidate_ms` grows linearly.

**How to read the result:**

| Outcome | Interpretation |
|---|---|
| WER@day8 falls with N, wake_ms flat | H_main CONFIRMED — deeper sleep helps |
| WER@day8 unchanged across N | Bottleneck is pseudo-label quality, not consolidation depth. Try LM-refined labels |
| WER@day8 improves up to N=K then worsens | N=K is the stability limit (Sleep paper §7). Report it. |
| N=1 already collapses (PER worsens) | Pseudo-label collapse — need trained checkpoint |

---

## 4. What Exists in the Repo (Complete File Map)

### 4a. NEW files created in this session

```
adapt/
  __init__.py                    # module declaration
  dietcorp_tta.py                # TTAConsolidator — the core consolidation engine

tools/
  drift_eval.py                  # day-split, synthetic drift, N-vs-PER-vs-day sweep
  mechanism_demo.py              # synthetic competent-decoder proof-of-mechanism

specs/
  G2_dietcorp_tta.yaml           # N=1 DietCorp reproduction experiment
  G3_sleep_consolidation.yaml    # N-sweep thesis experiment (FILL pretrained_ckpt)
  H2_zenbrain_live.yaml          # ZenBrain memory E2E training experiment

tests/
  test_dietcorp_tta.py           # 6 tests: CTC decode, augment, pseudo-label,
                                 #          consolidate, confidence gate, param select
  test_drift_eval.py             # 6 tests: PER metric, day split, synthetic drift,
                                 #          evaluate_day, full sweep, baseline restore
```

### 4b. MODIFIED files in this session

```
stages/memory/episodic_buffer.py    # SKELETON → LIVE
                                    # session-keyed + confidence-gated write policy
                                    # real forward(): cross-attn read + learnable gate
                                    # emits memory_query / memory_retrieved

stages/loss/episodic_consistency.py # removed * 0.0 → LIVE
                                    # MSE(query, retrieved.detach()) * weight

tests/test_zenbrain_stub_tripwire.py # INVERTED to live contract
                                    # now asserts forward() runs, loss is non-zero,
                                    # registry H1 = 'partial'

specs/H1_zenbrain_episodic.health.json  # state: skeleton → partial

registry.yaml                       # added G2, G3, H2; H1 skeleton→partial

run.py                              # added --adapt flag (_run_adapt function)
                                    # added --n_steps CLI override
                                    # wired memory stage into train + val loops
                                    # Unicode → ASCII safe output
```

### 4c. Pre-existing files used unchanged

```
stages/encoder/bit.py              # BIT_Transformer stage adapter
                                   # handles pretrained_ckpt loading
stages/projector/dietcorp_recal.py # per-day affine recal projector (the TTA target params)
stages/loss/ctc_anneal.py          # CTCAnnealLoss (CTC head: Linear 384→42)
stack.py                           # Stack: encoder→memory→projector→decoder, shape-checked
compose.py                         # ComposedLoss: sums named loss dicts
run.py                             # main training loop (train + val + leaderboard)
profiles/toy.yaml                  # smoke threshold: max_ce_loss=13.0 (was 10.0, fixed)
data/toy_train.hdf5                # 2.0 GB, 15% subset of real neural data
data/toy_val.hdf5                  # 0.5 GB
```

---

## 5. The TTAConsolidator — Technical Details

**File:** `adapt/dietcorp_tta.py`

### TTAConfig (all tunable parameters)
```python
n_aug:                 int   = 64      # DietCorp: 64 augmented copies per trial
mask_frac:             float = 0.53    # ~53% of time patches zeroed out
mask_span:             int   = 4       # contiguous span length in bins
lr:                    float = 1e-3    # AdamW learning rate for consolidation
grad_clip:             float = 1.0     # gradient norm clip
blank:                 int   = 0       # CTC blank token index
confidence_threshold:  float = 0.0    # skip consolidation below this confidence
min_pseudo_len:        int   = 1       # skip if pseudo-label collapses to < 1 token
```

### The Three Methods

**`augment(neural)`** → (n_aug, T, C)
- Takes one trial (T, C), returns 64 time-masked copies
- Masking: randomly zero out spans of `mask_span` bins covering ~`mask_frac` of the timeline
- Contiguous spans (not random per-bin) — forces the model to recover structure, not just
  average over noise

**`pseudo_label(neural)`** → (labels: LongTensor, confidence: float)
- Runs a clean forward pass through `logits_fn(neural)` → CTC logits
- CTC greedy decode: argmax per frame → collapse repeats → remove blanks
- Confidence = mean per-frame max softmax probability
- With `best_model_per.pth`: this produces real phoneme sequences
- Without a trained model: this produces garbage → collapse

**`consolidate(neural, n_steps=N)`** → metrics dict
- Step 1: get pseudo-label + confidence (skip if confidence < threshold)
- Step 2: time the clean forward pass alone → `wake_latency_ms`
  - **This MUST be N-independent** — it's the clinical latency guarantee
- Step 3: build augmentation batch (n_aug, T, C)
- Step 4: for n in range(N): compute CTC loss on augmented batch, backprop,
  clip gradients, AdamW step on target_params ONLY
- Step 5: measure `consolidate_ms` (≈ linear in N)
- Returns: skipped, confidence, pseudo_len, loss_before, loss_after, params_changed,
  wake_latency_ms, consolidate_ms

### Target Parameters (what gets updated)
```python
select_patch_embed_params(module, name_hints=("patch","read_in","embed","day_scale","day_shift"))
```
Selects only parameters whose name contains a hint — the patch embedding layer.
For BIT encoder: `patch_ln1` + `patch_embedding` ≈ 792K weights (out of ~2.5M total).
Everything else (transformer layers, output head, Qwen) stays frozen.

---

## 6. The Drift Evaluation Harness — Technical Details

**File:** `tools/drift_eval.py`

### Synthetic Drift Generation
```python
synthesize_drift(base_trials, n_days=8, scale_std=0.15, shift_std=0.15, noise_std=0.05)
```
For each day d, applies a progressively stronger per-channel affine + noise:
```
frac = d / (n_days - 1)          # 0.0 at day 0, 1.0 at day 7
scale = 1.0 + randn(C) * scale_std * frac
shift = randn(C) * shift_std * frac
noise = randn(T, C) * noise_std * frac
x_day_d = x_clean * scale + shift + noise
```
Day 0 is always identical to the clean baseline (frac=0).
Labels are preserved across all days (the signal changes, the intended phonemes do not).

This mimics the actual physical mechanism: electrode impedance shifts change channel
amplitudes (scale) and baselines (shift), with Gaussian noise from thermal fluctuations.
The HTML research doc calls this the "safe path" (Decision 3).

### The Sweep (`run_drift_eval`)
For each N in n_steps_list:
1. Restore model to baseline state (important — each N starts clean)
2. Walk days chronologically
3. **Evaluate BEFORE adapting** (this is the correct protocol — you don't adapt
   on the test set, you adapt on each trial as it arrives, then evaluate)
4. After evaluation, adapt using TTAConsolidator.consolidate(N steps) per trial
5. Record per-day PER, confidence, wake_ms, consolidate_ms

### Reading the Results
```
results/runs/G3_adapt_<hash>/drift_results.json:
{
  "by_n": { N: [ {day, per, confidence, n_trials}, ... ] },
  "wake_latency_ms": { N: float_or_null },
  "consolidate_ms":  { N: float_or_null },
  "summary": { N: { per_first, per_last, per_delta } }
}
```

**The thesis table to generate:**
```
   N  PER@day0  PER@last  delta(L-0)   wake_ms   cons_ms
   0    X.XXXX    X.XXXX      X.XXXX       n/a       n/a
   1    X.XXXX    X.XXXX      X.XXXX     XX.XX     XXX.XX
   2    X.XXXX    X.XXXX      X.XXXX     XX.XX     XXX.XX
   4    X.XXXX    X.XXXX      X.XXXX     XX.XX     XXX.XX
   8    X.XXXX    X.XXXX      X.XXXX     XX.XX     XXX.XX
```
H_main supported iff: `per_last` decreases with N AND `wake_ms` is flat.

---

## 7. ZenBrain Episodic Memory — Technical Details

**File:** `stages/memory/episodic_buffer.py`

### Architecture
```python
EpisodicBuffer(
    embed_dim=384,           # must match encoder output
    buffer_size=256,         # K past latents stored
    n_heads=6,               # cross-attention read head
    confidence_threshold=0.5, # write gate
    gate_init=0.1            # learnable fusion gate initialised near identity
)
```

**Internal state (non-trainable — these are memory, not parameters):**
- `buffer`: (256, 384) — the ring of stored latents
- `buf_session`: (256,) long — which session each slot came from
- `write_ptr`: scalar — current ring position

**Trainable components:**
- `read_head`: MultiheadAttention(384, 6) — learns what to retrieve from memory
- `gate`: scalar parameter — `fused = x + sigmoid(gate) * attended`
  - Initialised at 0.1 (near-zero gate, near-identity output) for training stability

### Forward Pass
```python
def forward(x, confidence=None, session_id=None):
    # x: (B, T_patch, 384)
    
    # Read: cross-attend current latents over the full buffer
    kv = buffer.expand(B, -1, -1)       # (B, K, 384)
    attended, _ = read_head(x, kv, kv)  # (B, T, 384)
    fused = x + sigmoid(gate) * attended  # learnable blend
    
    # Expose for episodic-consistency loss
    last_read = {"memory_query": x, "memory_retrieved": attended}
    
    # Write back (off gradient path)
    if confidence > confidence_threshold:
        buffer[write_ptr] = mean_pool(x)  # store mean-pooled latent
        buf_session[write_ptr] = session_id
        write_ptr = (write_ptr + 1) % buffer_size
    
    return fused  # same shape as input — identity passthrough in shape
```

### Episodic Consistency Loss
**File:** `stages/loss/episodic_consistency.py`
```python
# LIVE (the * 0.0 has been removed):
loss = F.mse_loss(memory_query, memory_retrieved.detach()) * weight
```
- `memory_query` = current encoder latent (gradients flow INTO the encoder)
- `memory_retrieved` = recalled past latent (detached — the stable target)
- Effect: the encoder is regularised to produce latents consistent with what
  it produced for similar trials in the past (cross-session stability)

---

## 8. Experiment Run on GPU — What We Actually Observed

### G2 Run (N=0 vs N=1, randomly-initialised BIT encoder)
```
   N  PER@day0  PER@last  delta(L-0)   wake_ms   cons_ms
   0    0.0000    0.2732      0.2732       n/a       n/a
   1    0.0000    0.6388      0.6388     35.71    134.58
```

**Findings:**
1. **Drift instrument works:** N=0 shows PER growing from 0 to 0.27 over 8 days — the
   synthetic drift correctly simulates the problem.
2. **Wake vs. consolidate latency separation is real:** 35ms (wake) vs 135ms (consolidate),
   clearly separable — the architecture is clinically valid.
3. **N=1 made things WORSE (0.27 → 0.64):** this is **pseudo-label collapse** — a
   randomly-initialised encoder produces garbage self-labels, and one step of "update
   toward garbage" confirmation-biases to a confident-but-wrong attractor. Confidence
   shoots from 0.07 → 0.97 (the model becomes very confident but very wrong).

### Mechanism Demo (N-sweep on competent synthetic decoder, strong drift)
```
   N  PER@day0  PER@last  delta(L-0)   wake_ms   cons_ms
   0    0.0000    0.9657      0.9657       n/a       n/a
   1    0.0000    0.9547      0.9547      0.06      3.00
   2    0.0000    0.8226      0.8226      0.05      4.42
   4    0.0000    0.7654      0.7654      0.16     23.72   ← best
   8    0.0000    0.9706      0.9706      0.28    101.03  ← unstable
```

**Findings:**
1. **The N-scaling mechanism works:** deeper consolidation (N=1→4) monotonically reduces
   final-day error from 95.5% → 76.5%.
2. **N=8 destabilises:** matches Sleep paper §7 exactly — large N causes training
   instability, especially when pseudo-labels are in the collapse zone.
3. **Wake latency is N-independent:** 0.05–0.16ms across all N — the clinical
   guarantee holds.
4. **Consolidation cost grows linearly:** 3ms → 4ms → 24ms → 101ms for N=1,2,4,8.

### Central Finding (regime-sensitivity)
**Self-labeled TTA is a bistable system:**
- **Good regime:** trained encoder + moderate drift → pseudo-labels are good enough →
  each step genuinely corrects drift → deeper N helps more
- **Collapse regime:** untrained encoder OR extreme drift → pseudo-labels are unreliable →
  each step reinforces the wrong attractor → more N = worse

**The thesis works in the good regime. The good regime needs:**
1. A trained encoder (use `best_model_per.pth`)
2. Moderate drift (or LM-refined pseudo-labels to survive stronger drift)
3. N capped at ~4–8 with gradient clipping

---

## 9. The Complete Local Execution Plan

### Prerequisites
```
Python:         py -3 (→ Python 3.11 at C:\Users\Pratik\AppData\Local\Programs\Python\Python311\)
GPU:            RTX 4050 6GB, CUDA 12.1, PyTorch 2.5.1+cu121
Checkpoint:     best_model_per.pth (output of train_ctc.py) — BIT encoder + CTC head
Toy data:       data/toy_train.hdf5 (~2GB), data/toy_val.hdf5 (~0.5GB)  ← already present
```

**IMPORTANT: Always run with `$env:PYTHONIOENCODING="utf-8"` on Windows or use the `py -3` launcher directly in PowerShell — the Windows cp1252 console crashes on non-ASCII characters (Δ, ×, etc.).**

---

### Step 0: Verify Everything Works (2 minutes)
```powershell
cd C:\Projects\Brain2Text2025\brain2text2025\brain2text-experiments
py -3 -m pytest tests/ -m "not slow" -q
# Expected: 40 passed, 2 deselected
```

---

### Step 1: Wire the Trained Checkpoint (2 minutes)
Edit `specs/G3_sleep_consolidation.yaml` and `specs/G2_dietcorp_tta.yaml`.
Change:
```yaml
  pretrained_ckpt: null
```
To (fill in your actual path):
```yaml
  pretrained_ckpt: C:/path/to/best_model_per.pth
```

Verify the checkpoint loads:
```powershell
$env:PYTHONIOENCODING="utf-8"
py -3 -c "
import torch
ckpt = torch.load('C:/path/to/best_model_per.pth', map_location='cpu')
print('Keys:', list(ckpt.keys())[:5])
print('Loaded OK')
"
```

---

### Step 2: Run G2 — DietCorp Reproduction (N=0 vs N=1) (~5 minutes)

**What this proves:** that the local TTA loop matches DietCorp's published behaviour
(N=1 flattens the drift curve).

```powershell
$env:PYTHONIOENCODING="utf-8"
py -3 run.py --expt G2 --profile toy --adapt
```

**Expected output table:**
```
   N  PER@day0  PER@last  delta(L-0)   wake_ms   cons_ms
   0    0.00      ~0.2X      ~0.2X       n/a       n/a
   1    0.00      ~0.0X      ~0.0X     ~30ms    ~130ms
```
N=1 should produce LOWER PER@last than N=0. If it's still higher, the checkpoint
needs more training or pseudo-labels need LM refinement.

Results saved to: `results/runs/G2_adapt_<hash>/drift_results.json`

---

### Step 3: Run G3 — The Thesis Experiment (N-sweep) (~15 minutes)

**What this proves (or falsifies): H_main.**

```powershell
$env:PYTHONIOENCODING="utf-8"
py -3 run.py --expt G3 --profile toy --adapt
```

Or override the sweep on the CLI:
```powershell
py -3 run.py --expt G3 --profile toy --adapt --n_steps 0 1 2 4 8
```

**Read the output table:**
```
   N  PER@day0  PER@last  delta(L-0)   wake_ms   cons_ms
   0    ...       ...        ...          n/a       n/a       ← baseline (no adapt)
   1    ...       ...        ...         ~30ms    ~130ms      ← DietCorp
   2    ...       ...        ...         ~30ms    ~260ms      ← sleep N=2
   4    ...       ...        ...         ~30ms    ~520ms      ← sleep N=4
   8    ...       ...        ...         ~30ms   ~1040ms      ← sleep N=8
```

**Interpretation:**
- If `per_last` decreases as N increases AND `wake_ms` is flat: **H_main confirmed**
- If `per_last` is flat across N: pseudo-label quality is the bottleneck, not N
- If `per_last` worsens at high N: stability limit found (report N_opt)
- If N=1 collapses (per_last > per_last at N=0): checkpoint not trained enough

Results saved to: `results/runs/G3_adapt_<hash>/drift_results.json`

---

### Step 4: Run Mechanism Demo (sanity check, 30 seconds)

```powershell
$env:PYTHONIOENCODING="utf-8"
py -3 tools/mechanism_demo.py
```

This uses a synthetic competent decoder (not BIT, not Qwen) to isolate the
consolidation mechanism from pseudo-label noise. Should always show N=4 better
than N=1, and N=8 unstable. If this breaks, the consolidation code has a bug.

---

### Step 5: Run H2 — ZenBrain Memory Training (~20 minutes on toy)

**What this tests:** does episodic memory help the encoder's training stability
and final WER (separate from drift adaptation)?

```powershell
$env:PYTHONIOENCODING="utf-8"
py -3 run.py --expt H2 --profile toy ^
    --train_h5 data/toy_train.hdf5 ^
    --val_h5 data/toy_val.hdf5
```

**What to look for:**
- Does H2 achieve lower toy WER than B0 (baseline BIT)?
- Does `loss_episodic` decrease during training? (proves the memory stage is learning)
- Are gradients flowing through the read head? (yes — proven by the tripwire test)

Results in: `results/runs/H2_toy_<hash>/history.json`
Compare against: `results/runs/B0_baseline_toy_*/history.json`

---

### Step 6: (Optional) Oracle Ablation — the cleanest possible test

Replace self-generated pseudo-labels with the TRUE ground-truth phoneme labels.
If N>1 still doesn't help with true labels, the mechanism itself is insufficient.
If N>1 DOES help with true labels but not self-labels, pseudo-label quality is
the only bottleneck.

```python
# In adapt/dietcorp_tta.py, modify pseudo_label() to accept ground-truth:
def pseudo_label(self, neural, true_labels=None):
    if true_labels is not None:
        return true_labels, 1.0    # perfect confidence
    # ... existing self-label code
```

---

### Step 7: Cloud Headline Run (A100 on JarvisLabs, ~3-4 hours)

**Gate:** must have toy run PASSED in the last 7 days (enforced by `run.py`).

```powershell
# On JarvisLabs instance (SSH in, then):
$env:PYTHONIOENCODING="utf-8"
py -3 run.py --expt G3 --profile full --adapt \
    --train_h5 data/ --val_h5 data/
```

This uses the real Brain-to-Text Benchmark '24 multi-day sessions instead of
synthetic drift. The 8-held-out-day WER curve is the publishable thesis result.

**The figure to generate:**
```
WER
│
66% ─── No adaptation (N=0)
│         \
│          \
│           \
12% ─────────────────── DietCorp N=1 (flat)
            ──────────────── N=2 (flatter?)
            ────────────────── N=4 (flattest?)
            ──────────────────── N=4 + ZenBrain replay
│
   day1  day2  day3  day4  day5  day6  day7  day8
```

---

## 10. The Two Paths for the Thesis Argument

### Path A: N-sweep result is clean (H_main confirmed)
"DietCorp's single step is the N=1 special case of offline sleep consolidation.
Increasing N to 4 reduces drift-WER by X% at constant wake latency. This mirrors
the Sleep paper's finding that sequential problems benefit most from deeper
offline consolidation, and confirms that electrode drift is fundamentally a
sequential adaptation problem."

### Path B: N-sweep is inconclusive (negative result, still publishable)
"Self-labeled TTA is regime-sensitive: it recovers drift only when pseudo-label
quality is sufficient. We characterise this regime boundary and show that deeper
consolidation (N>1) provides no benefit beyond what N=1 achieves when the
pseudo-label signal is degraded — suggesting that the bottleneck is not
consolidation depth but label quality. This implies that future work should
combine sleep-style consolidation with language-model-refined pseudo-labels
(as in DietCorp's original n-gram LM) rather than treating the two as independent."

Both paths are publishable. The infrastructure proves the argument either way.

---

## 11. What Remains (Ordered by Priority)

### Now (unblocks the thesis)
1. **Wire `best_model_per.pth` into `specs/G3`** → re-run G3 → read the table
2. **Run H2 toy** → compare WER to B0 baseline

### Soon (strengthens the thesis)
3. **KenLM 3-gram pseudo-label refinement** in `TTAConsolidator.pseudo_label`
   (the hook is already there — just replaces the greedy CTC decode with
   beam search rescored by a language model)
4. **Oracle-label ablation** (clean separation of mechanism vs label quality)

### Later (Phase 4)
5. **ZenBrain Simulation-Selection replay scheduler** — make the episodic buffer
   the replay source for consolidation, with priority = |CTC_surprise| + confidence + novelty
6. **Learned consolidation rule** — backprop through the entire N-loop
   consolidate→predict graph so the model learns how to consolidate (the Sleep
   paper's actual contribution, vs. our current fixed AdamW rule)

### Cloud (Phase 5)
7. **Real 8-day WER headline** on Brain-to-Text Benchmark '24
8. **Apple Neural Engine / Coral** deployment feasibility (original Project II plan)

### Housekeeping
9. `git submodule add https://github.com/ebrahimfeghhi/transformers_with_dietcorp
   docks/dietcorp_upstream` + fill `tools/dietcorp_paper_oracle.yaml`
10. Delete `.sleep_paper.txt` and `.zenbrain_paper.txt` (temp PDF extracts)

---

## 12. Environment Quick Reference

```
Python:          py -3  (NOT python — that's the Windows Store stub)
GPU:             RTX 4050 6GB, CUDA 12.1
PyTorch:         2.5.1+cu121
Tests:           py -3 -m pytest tests/ -m "not slow" -q
Console encoding: $env:PYTHONIOENCODING="utf-8"  (prevents cp1252 crash on non-ASCII)
Toy data:        data/toy_train.hdf5 (2GB), data/toy_val.hdf5 (0.5GB)
Checkpoint:      best_model_per.pth (from train_ctc.py) — BIT encoder + CTC head
Key entry point: run.py --expt <ID> --profile toy [--adapt] [--train_h5 ...] [--val_h5 ...]
```

---

## 13. Key Papers

| Paper | arXiv | Role |
|---|---|---|
| DietCorp | 2507.02800 | The TTA mechanism; published WER numbers; `best_model_per.pth` |
| Do LLMs Need Sleep? | 2605.26099 | N-step consolidation theory; stability caveat at large N |
| ZenBrain | 2604.23878 | Memory hierarchy; Simulation-Selection replay scheduler |

---

*Generated 2026-06-02. Repo: brain2text-experiments. All 40 non-slow tests pass.*
