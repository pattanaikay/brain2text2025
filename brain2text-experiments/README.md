# Brain2Text Experiments Framework

A modular, multi-track experimentation platform for training and evaluating brain-to-text neural pipelines. Designed to systematically explore encoder architectures, decoder variants, loss combinations, and projector designs through a structured registry of 25+ experiments.

## Overview

This framework implements a three-stage brain-to-text architecture:

```
Neural Input (ECoG) → Encoder → Projector → Decoder (LLM) → Text Output
```

Each component is modular and composable, allowing rapid experimentation across:
- **Track A**: Pretraining modality analysis (CKA alignment, perplexity, probing)
- **Track B**: Encoder architecture variants (Conformer, HRM, Mamba, MoE, ZenBrain)
- **Track C**: Decoder LLM variants (Qwen, Phi, Whisper-Qwen)
- **Track D**: Loss function ablations (CTC, contrastive, topological)
- **Track E**: Projector design variants (MLP, deep MLP, gated, QFormer)

**Baseline Performance**: 36.73% WER (BIT + Qwen2.5-1.5B)  
**Target**: 10% WER

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Shape Gate (CPU, ~10 sec)

Verify all architectures have compatible tensor shapes:

```bash
python -m pytest tests/test_stage_shapes.py -v
```

### 3. Toy Run (Local GPU, ~20 min)

Test the baseline (B0) on a small subset:

```bash
python run.py --expt B0_baseline --profile toy \
    --train_h5 data/toy_train.hdf5 \
    --val_h5   data/val.hdf5
```

### 4. Full Run (Cloud A100, ~2 hours)

After toy pass succeeds:

```bash
python run.py --expt B0_baseline --profile full \
    --train_h5 data/data_train.hdf5 \
    --val_h5   data/val.hdf5
```

## Core Components

### `run.py`
Main entrypoint for all experiments. Orchestrates:
1. Registry + spec loading
2. Profile override application (toy/full)
3. Stack construction and shape validation
4. Training loop with composed losses
5. Result logging to `leaderboard.sqlite`
6. Auto-pause on JarvisLabs after completion

**Usage:**
```bash
python run.py --expt <ID> --profile <toy|full> \
    --train_h5 <path> --val_h5 <path>
```

### `registry.yaml`
Single source of truth for all 25 experiments. Each entry includes:
- `track`: A, B, C, D, or E
- `name`: Human-readable title
- `description`: Scientific goal
- `spec_ref`: Path to YAML config
- `train_required`: Whether full training is needed
- `local_ok`: Whether experiment fits in 6 GB VRAM
- `expected_runtime_min`: Estimated duration
- `expected_wer_band`: [lo, hi] for post-run validation

**Example:**
```yaml
B1:
  track: B
  name: "Conformer Encoder"
  description: "Compare Conformer vs BIT baseline encoder"
  spec_ref: specs/B1_conformer.yaml
  train_required: true
  local_ok: true
  expected_runtime_min: 25
  expected_wer_band: [0.25, 0.45]
```

### `stack.py`
Runtime builder for the neural pipeline. Loads stages dynamically:

```python
stack = Stack(spec)
# stack["encoder"]   : BIT, Conformer, HRM, Mamba, MoE, ZenBrain
# stack["projector"]  : MLP, Deep MLP, Gated, QFormer
# stack["decoder"]    : Qwen1.5, Qwen2-Audio, Whisper-Qwen
```

Shape contracts are validated at build time:
```
encoder output  : (T_patch, 384)
projector output: (T_out, llm_dim)  ← automatically determined from decoder
decoder         : (vocab_size,)
```

### `compose.py`
Multi-loss composition system. Combines losses via forward hooks:

```python
loss_fns = compose([
    build_ce({}, prev_shape=None)[0],
    build_ctc({"anneal_epochs": 75}, prev_shape=None)[0],
    build_contrastive({"weight": 1.0}, prev_shape=None)[0],
])

total_loss, breakdown = loss_fns(batch, stack, outputs)
# breakdown = {"loss_ce": ..., "loss_ctc": ..., "loss_contrastive": ...}
```

## Project Structure

```
brain2text-experiments/
├── run.py                    # Main entrypoint
├── registry.yaml             # Experiment registry
├── compose.py                # Multi-loss composition
├── stack.py                  # Pipeline builder
├── requirements.txt          # Dependencies
├── start-commands.txt        # Quick reference commands
│
├── specs/                    # Experiment YAML configs
│   ├── A1_cka.yaml          # Track A: modality analysis
│   ├── B0_bit_scratch.yaml  # Track B: encoder variants
│   ├── C3_whisper_qwen.yaml # Track C: decoder variants
│   ├── D1b_ctc_anneal.yaml  # Track D: loss ablations
│   └── E1a_deep_mlp.yaml    # Track E: projector variants
│
├── stages/                   # Modular stage implementations
│   ├── encoder/             # BIT, Conformer, HRM, Mamba, MoE, ZenBrain
│   ├── projector/           # MLP, Deep MLP, Gated, QFormer
│   ├── decoder/             # Qwen1.5, Qwen2-Audio, Whisper-Qwen
│   └── loss/                # CE, CTC, Contrastive, Topological
│
├── tests/                    # Test suite
│   └── test_stage_shapes.py # Shape validation (CPU)
│
├── tools/                    # Utilities
│   ├── make_toy_dataset.py  # Generate toy subset
│   └── cka_analysis.py      # CKA embedding alignment
│
├── profiles/                 # Execution profiles
│   ├── toy.yaml             # Local RTX 4050 (~20 min)
│   └── full.yaml            # Cloud A100 (150 epochs)
│
├── results/                  # Output storage
│   ├── leaderboard.sqlite    # Training metrics
│   ├── plot_tracks.py        # Visualization
│   └── <expt_id>/            # Per-experiment outputs
│       ├── best_wer.txt
│       ├── metrics.json
│       └── model.pth
│
├── cache/                    # Intermediate files
└── docks/                    # Docker configs (optional)
```

## Experiments Overview

### Track A: Pretraining Modality Study (Analysis, No Training)

Determines which LLM backbone embedding space aligns best with the BIT encoder.

| ID  | Name | Task | Runtime |
|-----|------|------|---------|
| A1  | CKA Embedding Alignment | Compute CKA similarity between encoder and 3 LLM spaces | 30 min |
| A2  | Spoken Language Perplexity | Compare audio-pretrained vs text-only LLM perplexity | 15 min |
| A3  | Phoneme Probing Accuracy | Linear probe on frozen LLM hidden states | 45 min |
| A4  | Audio vs Vision E2E | Full E2E: text vs audio vs vision pretrained LLM | 60 min |

**Run A-track:**
```bash
python tools/cka_analysis.py --val_h5 data/val.hdf5 --out results/cka_results.json
```

### Track B: Encoder Variants (E2E Training)

Compares different neural encoder architectures while keeping projector and decoder fixed.

| ID | Name | Architecture | Parameters | VRAM | Local |
|----|------|--------------|------------|------|-------|
| B0 | BIT Scratch | BIT | 12M | 3.2 GB | ✓ |
| B1 | Conformer | Conformer | 28M | 4.1 GB | ✓ |
| B2 | HRM | Hierarchical RNN | 18M | 3.8 GB | ✓ |
| B3 | Mamba/GRU | Mamba or GRU fallback | 24M | 4.0 GB | ✓ |
| B4 | MoE | Mixture of Experts | 32M | 4.5 GB | ✓ |
| B5 | ZenBrain | ZenBrain backbone | 16M | 3.6 GB | ✓ |

**Dependency**: B0 baseline must pass toy run before B1-B5 allowed to run full.

```bash
# Baseline first
python run.py --expt B0_baseline --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

# Then any B1-B5
python run.py --expt B1 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
```

### Track C: Decoder LLM Variants (E2E Training)

Compares different language models as the decoder component.

| ID | Name | Model | Size | VRAM | Local |
|----|------|-------|------|------|-------|
| C1 | Qwen2-Audio | Qwen2-7B-Instruct-Audio | 7B | 16 GB | ✗ |
| C2 | Phi-4MM | Phi-4-Multimodal | 5.6B | 13 GB | ✗ |
| C3 | Whisper-Qwen | Whisper + Qwen1.5 | 4.1B | 4.1 GB | ✓ |

**Note**: C1 and C2 exceed 6 GB VRAM; run on cloud only.

```bash
# C3 runs locally
python run.py --expt C3 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

# C1, C2 cloud only (skip locally)
```

### Track D: Loss Function Ablations (E2E Training)

Isolates the contribution of individual loss components using the same architecture.

| ID | Name | Configuration | Notes |
|----|------|---------------|-------|
| D1a | CTC Baseline | CTC only, no anneal | Baseline loss |
| D1b | CTC + Anneal | CTC linear anneal over 75 epochs | Smooth transition |
| D1d | No CTC | CE only | Ablate CTC |
| D2a | No Contrastive | CE + CTC, no contrastive | Ablate contrastive |
| D2d | Contrastive ×2 | Contrastive weight × 2 | Increase contrastive |
| D3a | Topo 0.0001 | Topological λ=0.0001 | Light regularization |
| D3b | Topo 0.001 | Topological λ=0.001 | Medium regularization |
| D3c | Topo 0.01 | Topological λ=0.01 | Heavy regularization |
| D4 | Label Smoothing | Label smoothing ε=0.1 | Regularization variant |

All D experiments use identical architecture; results isolate loss impact.

```bash
# All D experiments can run in parallel (same architecture)
for expt in D1b D1d D2a D2d D3b D3c D4; do
  python run.py --expt $expt --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5 &
done
wait
```

### Track E: Projector Design Variants (E2E Training)

Explores different projection layers between encoder and decoder embeddings.

| ID | Name | Architecture | Parameters | Local |
|----|------|--------------|------------|-------|
| E1a | Deep MLP | 3-layer MLP (512→256→128) | 1.2M | ✓ |
| E1b | Gated MLP | Gated variant of MLP | 1.1M | ✓ |
| E2b | QFormer 32 | QFormer with 32 queries | 8.5M | ✓ |
| E3 | Patch QFormer | QFormer with patch grid | 9.2M | ✓ |

**Dependency**: E2b must pass before E3 can run.

```bash
# E2b first (E3 depends on it)
python run.py --expt E2b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

# Then E3
python run.py --expt E3 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
```

## Execution Flow

### The Mandatory Three-Step Progression

Every experiment must follow this sequence; `run.py` enforces it:

```
pytest (CPU) → toy run (local GPU, ~20 min) → full run (A100, 150 epochs)
```

**The full run hard-blocks if no toy pass exists in `leaderboard.sqlite` within the last 7 days.**

### Step 1: Shape Gate (CPU, ~10 sec)

```bash
python -m pytest tests/test_stage_shapes.py -v
```

Catches tensor shape mismatches in every encoder/projector **before** touching a GPU. Run once before any experiments.

**Output**: Pass/fail per encoder-projector-decoder combination.

### Step 2: Toy Runs (Local GPU, ~20 min each)

Train on toy dataset (5% of full data) to validate experimental design:

```bash
python run.py --expt <ID> --profile toy \
    --train_h5 data/toy_train.hdf5 \
    --val_h5   data/val.hdf5
```

**What toy validates:**
- Architecture compatibility (no OOM, shape errors)
- Loss computation (no NaN, divergence)
- Data pipeline (no data corruption)
- Metrics logging (results saved to leaderboard.sqlite)

**Output**:
- `results/<expt_id>/best_wer.txt`: Best WER from toy run
- `results/<expt_id>/metrics.json`: Full metrics log
- `results/leaderboard.sqlite`: Row added for this (expt, toy) run

### Step 3: Full Runs (Cloud A100, 150 epochs, ~2 hours)

Only allowed if toy pass exists within 7 days:

```bash
python run.py --expt <ID> --profile full \
    --train_h5 data/data_train.hdf5 \
    --val_h5   data/val.hdf5
```

Automatically pauses JarvisLabs instance after completion.

**Output**:
- `results/<expt_id>/best_wer.txt`: Final WER
- `results/<expt_id>/model.pth`: Checkpoint
- `results/leaderboard.sqlite`: Full row with all metrics

## Profiles

Profiles override spec YAML defaults for local vs. cloud execution.

### `profiles/toy.yaml`
- **Hardware**: Local RTX 4050 (6 GB VRAM)
- **Batch size**: 8
- **Epochs**: 3 (quick validation)
- **Duration**: ~20 min
- **Purpose**: Catch bugs, validate architecture, gate full runs
- **WER threshold**: Must be within `expected_wer_band` from registry

### `profiles/full.yaml`
- **Hardware**: JarvisLabs A100 (80 GB VRAM)
- **Batch size**: 32
- **Epochs**: 150 (production training)
- **Duration**: ~2 hours
- **Purpose**: Final benchmarks, hyperparameter tuning
- **Auto-pause**: Stops instance after training completes

## Results & Leaderboard

Results are centralized in `results/leaderboard.sqlite`:

```bash
# List all results
python results/leaderboard.py --list

# Filter by track
python results/leaderboard.py --track B

# Plot comparison
python results/plot_tracks.py --track B --profile toy
```

**Columns in leaderboard.sqlite**:
- `expt_id`: Experiment ID (e.g., "B1")
- `track`: Track (A, B, C, D, E)
- `profile`: Profile used (toy, full)
- `best_wer`: Best WER achieved
- `wer_epoch`: Epoch where best WER occurred
- `slope`: WER improvement slope (epochs 50-100)
- `timestamp`: When run completed
- `spec_hash`: Config file hash (for reproducibility)
- `code_hash`: Source code hash

## Installation & Setup

### System Requirements
- **CPU**: x86-64
- **GPU**: NVIDIA (torch + CUDA 11.8+)
- **RAM**: 8 GB minimum (16 GB recommended)
- **Storage**: 100 GB minimum (data + models + outputs)

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Core packages**:
- `torch>=2.5.1`: Deep learning framework
- `transformers>=4.45.0`: Pretrained models
- `peft>=0.13.0`: Parameter-efficient finetuning
- `bitsandbytes>=0.44.0`: 8-bit optimization
- `h5py`: HDF5 data loading
- `PyYAML`: Config parsing

**Optional for specific architectures**:
```bash
# Mamba encoder (B3_mamba_ssm)
pip install mamba-ssm causal-conv1d

# QFormer projector (E2b, E3)
pip install qformer-transformers
```

### Create Toy Dataset

```bash
python tools/make_toy_dataset.py \
    --full_path data/data_train.hdf5 \
    --toy_path data/toy_train.hdf5
```

Creates a 5% subset for fast local testing.

## Common Commands

### Run Shape Tests
```bash
python -m pytest tests/test_stage_shapes.py -v
```

### Run Single Experiment (Toy)
```bash
python run.py --expt B1 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
```

### Run Experiment (Full)
```bash
python run.py --expt B1 --profile full --train_h5 data/data_train.hdf5 --val_h5 data/val.hdf5
```

### Run Multiple Experiments (Parallel)
```bash
for expt in B1 B2 B3 B4 B5; do
  python run.py --expt $expt --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5 &
done
wait
```

### Analyze Results
```bash
python results/leaderboard.py --list
python results/plot_tracks.py --track B
```

### View Experiment Config
```bash
cat specs/B1_conformer.yaml
```

## Architecture Details

### Stage API

Every stage (encoder, projector, decoder, loss) exports:

```python
def build(spec: dict, prev_shape: tuple) -> tuple[nn.Module, tuple]:
    """
    spec: Configuration dict from YAML
    prev_shape: Input tensor shape (non-batch), e.g., (T, 384)
    
    Returns:
        (nn.Module, output_shape)  # e.g., (encoder, (T_out, 384))
    """
```

### Encoder Output Shape
All encoders output shape `(T_patch, 384)` where:
- `T_patch`: Number of time patches (~200 for 2-second window)
- `384`: Fixed hidden dimension for projector compatibility

### Projector Output Shape
Adapts to decoder; typically `(T_out, llm_dim)` where:
- `llm_dim`: Decoder embedding dimension (e.g., 1280 for Qwen1.5-1.5B)
- Validated at Stack build time

### Loss Composition

Forward hooks allow dynamic loss combinations without retraining:

```python
# Original trained stack
stack = Stack(spec)

# Apply new loss combination
loss_fns = compose([build_ce(), build_ctc()])

# Evaluate same encoder under different losses (Track D ablations)
for batch in val_loader:
    outputs = stack(batch)
    total_loss, breakdown = loss_fns(batch, stack, outputs)
```

## Troubleshooting

### "Shape mismatch: encoder output is (T, 384), projector input expects (T, 256)"

Run shape tests to identify incompatibility:
```bash
python -m pytest tests/test_stage_shapes.py -v
```

Fix by updating the projector spec in `specs/` to accept (T, 384).

### "No toy pass in leaderboard.sqlite; full run blocked"

Run toy first:
```bash
python run.py --expt <ID> --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
```

### "CUDA out of memory (OOM)"

- Use `--profile toy` to reduce batch size
- Or run on cloud (JarvisLabs A100) if `local_ok: false` in registry

### "Mamba encoder not found"

Install the C extension:
```bash
pip install mamba-ssm causal-conv1d
```

Requires CUDA 11.8+. Fall back to B3 (GRU) if unavailable.

## File Descriptions

| File | Purpose |
|------|---------|
| `run.py` | Main training entrypoint; enforces toy→full progression |
| `registry.yaml` | Experiment definitions; single source of truth |
| `stack.py` | Pipeline builder; validates shape contracts |
| `compose.py` | Multi-loss hook system for ablations |
| `specs/*.yaml` | Per-experiment configuration |
| `stages/` | Modular encoder/projector/decoder/loss implementations |
| `tests/test_stage_shapes.py` | CPU-only shape validation |
| `tools/make_toy_dataset.py` | Create 5% subset for local testing |
| `tools/cka_analysis.py` | Compute CKA alignment scores (Track A) |
| `profiles/*.yaml` | Execution profiles (toy/full) |
| `results/leaderboard.sqlite` | Centralized results database |
| `results/leaderboard.py` | Query leaderboard |
| `results/plot_tracks.py` | Visualize experiment results |

## Contributing

When adding new experiments:

1. **Create spec YAML** in `specs/`
2. **Add registry entry** in `registry.yaml`
3. **Implement stage** in `stages/` (if new variant)
4. **Update tests** in `tests/` if shapes change
5. **Run shape tests**: `pytest tests/test_stage_shapes.py -v`
6. **Run toy**: `python run.py --expt <ID> --profile toy ...`
7. **Commit**: Include spec hash in commit message for reproducibility

## References

- **Baseline**: BIT encoder + Qwen2.5-1.5B decoder (36.73% WER)
- **Target**: 10% WER
- **Data**: ECoG neural recordings (T=2 sec, 128 channels)
- **Evaluation**: Word Error Rate (WER) on validation set

---

**Last Updated**: May 2026  
**Framework Version**: 1.0  
**Status**: Active (25 experiments, expanding)
