# Brain-to-Text 2025: End-to-End Neural Decoding with BIT (Brain-to-Text Integration Transformer)

This repository implements the **BIT Framework** — a complete end-to-end neural decoding system that converts brain neural activity into natural English text. The model combines a Transformer-based neural encoder with a fine-tuned language model (Qwen2.5-1.5B-Instruct) using advanced alignment techniques.

## 🎯 Overview

### Architecture
The BIT framework consists of three integrated components:

```
Raw Neural Data (512 channels, 20ms bins)
    ↓
[BIT_Transformer Encoder] → Session-specific read-in + Temporal Patching + RoPE Attention
    ↓
Neural Tokens (384-dim, 100ms patches)
    ↓
[MLPProjector] → Modality alignment bridge
    ↓
LLM Embeddings (1536-dim)
    ↓
[Qwen2.5-1.5B with LoRA + 4-bit Quantization] → Language generation
    ↓
English Text Output
```

### Key Features
- **Session-Specific Read-in**: Per-session linear transformation layers to mitigate electrode drift across recording days
- **Temporal Patching**: Aggregates 5×20ms bins (100ms context) to reduce sequence length while preserving temporal dynamics
- **Rotary Positional Embeddings (RoPE)**: Modern position encoding applied at each transformer block for better long-range attention
- **Multimodal Alignment Loss**: InfoNCE contrastive loss ($\mathcal{L}_{contrastive}$) aligns neural and text embeddings in shared latent space
- **4-bit Quantization + LoRA**: Enables fine-tuning of 1.5B parameter LLM on consumer GPUs
- **Careful Label Masking**: Prevents prompt memorization by masking instruction text from loss computation
- **End-to-End Training**: Joint optimization of cross-entropy loss and contrastive loss ($\mathcal{L}_{total} = \mathcal{L}_{CE} + \mathcal{L}_{contrastive}$)

## 📁 Project Structure

```
brain2text-modeltraining/
├── COMPREHENSIVE_TECHNICAL_SPEC.md   # Deep-dive into every component, layer, and formula
├── debug_e2e_training.md             # Troubleshooting guide for common training issues
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── src/
│   ├── models/
│   │   ├── encoder.py                # BIT_Transformer: Session-specific encoder with RoPE
│   │   ├── projector.py              # MLPProjector: Neural→LLM embedding bridge (3 layers)
│   │   └── bit_e2e.py                # BrainToTextE2E: Complete end-to-end model
│   │
│   ├── preprocessing/
│   │   ├── dataloader.py             # Preprocessed_BCI_Dataset: Multi-file HDF5 loader
│   │   │                             # with Z-score normalization and Gaussian smoothing
│   │   └── compute_session_stats.py  # Compute per-session statistics for normalization
│   │
│   └── utils/
│       ├── metrics.py                # WER, CER evaluation metrics
│       └── logging_utils.py          # Logging setup and experiment tracking
│
├── scripts/
│   ├── train_e2e.py                  # Main training script for end-to-end fine-tuning
│   ├── train_e2e_local.py            # Local training variant (debugging)
│   ├── evaluate.py                   # Generate predictions and compute metrics
│   ├── plot_metrics.py               # Visualize training curves (loss, WER, CER)
│   ├── diagnostic_ctc.py             # Debug CTC-related issues
│   ├── dry_run.py                    # Quick validation of model setup
│   └── create_dummy_data.py          # Generate synthetic data for testing
│
├── outputs/                          # Training logs, histories, predictions
├── local_checkpoints/                # Model checkpoints during training
└── data/                             # Link to HDF5 data files
```

## 🧠 Core Components

### 1. BIT_Transformer (Neural Encoder) - `encoder.py`

Converts raw 512-channel electrode signals into semantic neural tokens.

| Component | Config | Purpose |
|-----------|--------|---------|
| **Input** | `[B, T, 512]` | Raw neural signal (B batches, T time steps, 512 electrode channels) |
| **Session Read-in** | `nn.ModuleDict[session_id → Linear(512, 512)]` | Per-session channel transformation (mitigates electrode drift) |
| **Patch Embedding** | `Linear(2560, 384)` | Aggregates 5×20ms bins → 1×100ms patch, projects to 384-dim |
| **RoPE Attention** | 7 transformer layers, 6 heads, 384-dim | Rotary position embeddings for efficient long-range attention |
| **Output** | `[B, T', 384]` | Neural tokens (T' = ceil(T/5) due to patching) |

**Key Design Decision**: Session-specific read-in layers allow the encoder to adapt to electrode drift, a major source of error in neural recordings. Instead of forcing all sessions through a shared transformation, we learn a linear mapping per session ID.

### 2. MLPProjector (Modality Bridge) - `projector.py`

Maps neural embeddings (384-dim) to LLM input space (1536-dim).

```python
nn.Sequential(
    nn.Linear(384, 1024),      # Expand: 384 → 1024
    nn.ReLU(),
    nn.Linear(1024, 1024),     # Hidden: maintain 1024-dim
    nn.ReLU(),
    nn.Linear(1024, 1536),     # Project to LLM space: 1024 → 1536
    nn.LayerNorm(1536)         # Stabilize output
)
```

**Parameters**: ~3.2M (0.2% of LLM)

### 3. ModalityAlignmentLoss (Contrastive Learning) - `bit_e2e.py`

InfoNCE loss aligning neural and text representations:

$$\mathcal{L}_{contrastive} = \frac{1}{2} \left( \mathcal{L}_n + \mathcal{L}_t \right)$$

where:
- $\mathcal{L}_n = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ (neural → text)
- $\mathcal{L}_t = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)}$ (text → neural)
- $s_{ij} = \text{normalize}(\text{neural}_i) \cdot \text{normalize}(\text{text}_j)$
- $\tau$ is a learnable temperature parameter (clamped $\geq 10^{-4}$)

This forces the model to learn representations where neural and text embeddings are close in the shared space.

### 4. BrainToTextE2E (Full Model) - `bit_e2e.py`

**Components**:
- `self.llm`: Qwen2.5-1.5B-Instruct with 4-bit quantization and LoRA
- `self.neural_encoder`: BIT_Transformer
- `self.projector`: MLPProjector
- `self.contrastive_loss_fn`: ModalityAlignmentLoss

**LLM Configuration**:
- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization**: 4-bit NF4 (bitsandbytes) + double quantization
- **LoRA Targets**: `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
- **LoRA Params**: r=8, α=32, dropout=0.2 (following Zhang et al. 2025)

### 5. Preprocessed_BCI_Dataset (Data Loading) - `dataloader.py`

Handles multi-file HDF5 loading with preprocessing:

**Processing Pipeline**:
1. **Z-score Normalization**: Per-session mean/std from `session_stats.json`
2. **Gaussian Smoothing**: σ=1.5 along temporal axis
3. **Temporal Patching**: Aggregates consecutive time steps
4. **CTC Validation** (optional): Filters trials where patched_length < phoneme_length
5. **Collation**: Pads sequences to batch max length

## 📊 Training Pipeline

### Forward Pass (Training)

```
1. Neural Encoding:
   neural_data[B, T, 512] → BIT_Transformer → neural_tokens[B, T', 384]

2. Projection:
   neural_tokens → MLPProjector → projected_embeds[B, T', 1536]

3. Prompt Construction:
   [PROMPT_START] + [NEURAL_TOKENS] + [PROMPT_END] + [TARGET_TEXT]
   
   Where PROMPT_START = "<|im_start|>user\n<neural_activity>\n"
         PROMPT_END   = "\n</neural_activity>\ndecode into English:<|im_end|>\n<|im_start|>assistant\n"

4. Label Masking (KEY FIX):
   full_labels initialization: all -100 (ignored by loss)
   full_labels[prefix_len:] = target_token_ids (trained)
   
   This ensures the model ONLY learns from target text, not instructions!

5. LLM Forward:
   outputs = LLM(inputs_embeds=full_embeds, attention_mask=mask, labels=full_labels)
   CE_loss = outputs.loss

6. Contrastive Loss:
   neural_pooled = mean(projected_embeds)
   text_pooled = masked_mean(text_embeds)
   contrastive_loss = ModalityAlignmentLoss(neural_pooled, text_pooled)

7. Total Loss:
   loss = CE_loss + contrastive_loss
```

### Key Implementation Details

**Label Masking Strategy** (prevents prompt memorization):
- Prompt tokens, neural tokens, padding: **-100** (loss ignored)
- Target text tokens: **actual token IDs** (loss computed)
- Ensures model learns from neural signals, not instructions

**Attention Mask Construction**:
- 2D mask: `[batch_size, seq_len]`
- Masks padded neural positions (based on `neural_lengths`)
- Masks padded text positions (based on tokenizer attention_mask)
- HuggingFace LLM applies causal mask internally

**Diagnostic Logging**:
- First training batch: prints predicted vs. target tokens
- Helps verify the model isn't just memorizing text
- Disabled after first batch to avoid log spam

## 🚀 Usage

### Installation

```bash
pip install -r requirements.txt
```

### 0. (Optional) Quick Validation

```bash
python scripts/dry_run.py
```

Initializes the model and runs a single forward pass to verify setup.

### 1. Prepare Data

Compute session statistics for Z-score normalization:

```bash
python src/preprocessing/compute_session_stats.py \
    --h5_list data/h5_list.json \
    --output_path session_stats.json
```

**Output**: `session_stats.json` with per-session mean and std for each electrode.

### 2. Train End-to-End

```bash
python scripts/train_e2e.py \
    --train_h5 /path/to/train/data/ \
    --val_h5 /path/to/val/data/ \
    --session_stats session_stats.json \
    --output_dir outputs/e2e_run1 \
    --epochs 100 \
    --batch_size 4 \
    --lr 1e-4 \
    --no_quantize  # Remove for 4-bit quantization
```

**Key Arguments**:
- `--train_h5`, `--val_h5`: Path(s) to HDF5 files or directories (auto-discovers `data_train.hdf5`, `data_val.hdf5`)
- `--session_stats`: JSON file with per-session normalization stats
- `--pretrained_encoder`: (Optional) Load checkpoint from SSL pretraining
- `--patch_size`: Temporal patch size (default: 4, i.e., 5 bins → 1 patch)
- `--no_quantize`: Disable 4-bit quantization (uses full precision)

**Outputs**:
- `outputs/e2e_run1/checkpoints/`: Model checkpoints
- `outputs/e2e_run1/training_history.json`: Loss/metric logs
- `outputs/e2e_run1/e2e_train.log`: Detailed training logs

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --test_h5 /path/to/test.hdf5 \
    --checkpoint outputs/e2e_run1/checkpoints/best_wer.pth \
    --session_stats session_stats.json \
    --output_dir outputs/e2e_run1/predictions
```

**Outputs**:
- `predictions.json`: Model predictions
- `metrics.json`: WER, CER scores

### 4. Visualize Training

```bash
python scripts/plot_metrics.py \
    --history outputs/e2e_run1/training_history.json \
    --output_dir outputs/e2e_run1/plots
```

Generates plots for:
- Training/validation loss
- WER and CER over time
- Learning rate schedule

## 🔧 Advanced Configuration

### Custom Hyperparameters

Edit `train_e2e.py` defaults or pass via CLI:

```bash
python scripts/train_e2e.py \
    --train_h5 data/ \
    --batch_size 8 \
    --lr 5e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --patch_size 4 \
    --dropout 0.15
```

### Loading Pretrained Encoder

```bash
python scripts/train_e2e.py \
    --train_h5 data/ \
    --pretrained_encoder checkpoints/ssl_best.pth \
    --freeze_encoder_epochs 5  # Freeze encoder for first N epochs
```

### Resume Training

```bash
python scripts/train_e2e.py \
    --train_h5 data/ \
    --resume_from_checkpoint outputs/e2e_run1/checkpoints/latest.pth
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `CUDA out of memory`
- **Solution**: Reduce `--batch_size`, enable `--no_quantize=False` (default is 4-bit quantization), or use gradient accumulation

**Issue**: Low loss but poor predictions (label leakage)
- **Solution**: Check `debug_e2e_training.md` for masking verification. Ensure `full_labels` has -100 for prompt portion.

**Issue**: High loss with random predictions (model not learning)
- **Solution**: Verify data loading with `python scripts/create_dummy_data.py`. Check logs for encoding errors. Ensure session_stats matches data.

**Issue**: OOM during generation
- **Solution**: Reduce `--max_new_tokens` in `generate()` call, or decrease batch size

See [debug_e2e_training.md](debug_e2e_training.md) for detailed debugging.

## 📈 Expected Performance

### Phase 1: E2E Training
- **Initial CE Loss**: ~5-8 (first 10 epochs)
- **Final CE Loss**: ~2-4 (after 50+ epochs)
- **Initial WER**: ~100% (random)
- **Final WER**: ~30-40% (competitive submission)
- **Contrastive Loss**: Decreases with proper alignment

### Metrics
- **WER**: Word Error Rate (Levenshtein distance between predicted and target)
- **CER**: Character Error Rate (character-level Levenshtein distance)
- **Logs**: Comprehensive training logs in `outputs/*/logs/`

## 📚 Model Architecture Details

### BIT_Transformer Architecture Table

| Layer | Type | Input → Output | Parameters | Notes |
|-------|------|---|---|---|
| **Read-in** | `nn.ModuleDict[Linear]` | `[B, T, 512] → [B, T, 512]` | 512×512×N_sessions | Per-session linear (drift correction) |
| **Patch Embedding** | `nn.Linear` | `[B, T', 2560] → [B, T', 384]` | ~985K | Aggregates 5 bins, projects to embedding |
| **Transformer Block (×7)** | `TransformerEncoder` | `[B, T', 384] → [B, T', 384]` | 6 multi-head atts + 7 FFNs | RoPE attention, 6 heads, 384-dim |
| **LayerNorm** | `nn.LayerNorm` | `[B, T', 384] → [B, T', 384]` | 768 (γ + β) | Output normalization |
| **Total Encoder** | - | - | **~3.2M** | - |

### MLPProjector Architecture Table

| Layer | Input → Output | Parameters | Activation |
|-------|---|---|---|
| Linear 1 | 384 → 1024 | 394.2K | ReLU |
| Linear 2 | 1024 → 1024 | 1.049M | ReLU |
| Linear 3 | 1024 → 1536 | 1.573M | - |
| LayerNorm | 1536 → 1536 | 3.072K | - |
| **Total Projector** | - | **~3.02M** | - |

### Full Model Parameter Count

| Component | Parameters | Trainable | Notes |
|-----------|---|---|---|
| BIT_Transformer | 3.2M | ✅ Full | Neural encoder |
| MLPProjector | 3.02M | ✅ Full | Modality bridge |
| Qwen2.5-1.5B LLM | 1.5B | ✅ LoRA only | 7 target modules, r=8 |
| LoRA Weights | ~1.8M | ✅ Full | q,k,v,o,gate,up,down projections |
| **Total Trainable** | **~1.51B** | - | Including LoRA |
| **Frozen LLM weights** | **~1.498B** | ❌ | 4-bit quantized base |

## 📐 Mathematical Formulation

### Cross-Entropy Loss (LLM)

$$\mathcal{L}_{CE} = -\sum_{t=1}^{T_{target}} \mathbb{1}[\text{labels}_t \neq -100] \log p(y_t | y_{<t}, \text{neural})$$

where:
- $\mathbb{1}[\cdot]$ masks out ignored tokens (prompt, padding)
- $p(y_t)$ is LLM's predicted probability distribution
- Sum only over valid target tokens

### Contrastive Loss (InfoNCE)

$$\mathcal{L}_{contrastive} = \frac{1}{2} \left( \mathcal{L}_n + \mathcal{L}_t \right)$$

$$\mathcal{L}_n = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}, \quad \mathcal{L}_t = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)}$$

where:
- $s_{ij} = \text{norm}(z_i^{neural}) \cdot \text{norm}(z_j^{text})$ (cosine similarity)
- $\tau$ is learnable temperature

### Total Loss

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{contrastive}$$

(Default: $\lambda = 1.0$)

## 🏗️ Training Dynamics

### Phase 1: Early Training (Epochs 1-20)
- **Behavior**: Model learns basic text generation from neural features
- **Loss Profile**: CE loss ~5-8, contrastive loss ~0.5-1.0
- **Diagnostics**: Check diagnostic logs to verify predicted tokens improve

### Phase 2: Mid Training (Epochs 20-60)
- **Behavior**: Encoder learns more complex neural patterns, LLM refines language
- **Loss Profile**: CE loss ~3-5, contrastive loss ~0.1-0.3
- **Diagnostics**: WER should drop from 80-90% → 40-60%

### Phase 3: Late Training (Epochs 60+)
- **Behavior**: Fine-grained pattern learning, overfitting risk
- **Loss Profile**: CE loss ~2-4, contrastive loss ~0.01-0.1
- **Diagnostics**: Monitor validation WER for plateau/increase; checkpoint best model

### Learning Rate Schedule
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10 epochs)
- **Warmup**: Optional ~500 steps linear warmup

## 🔑 Critical Implementation Details

### Label Masking (Prevents Prompt Memorization)

This is the most important fix in the current implementation. Without proper masking:
- Model learns to generate text from **instructions**, not from **neural signals**
- WER stays high (~100%)
- Cross-entropy loss seems low but model learns nothing useful

**Solution**: Initialize all labels to -100 (ignore), then fill in only target text:

```python
full_labels = torch.full((batch_size, seq_len), -100, device=device, dtype=torch.long)
full_labels[:, prefix_len:] = target_token_ids  # Only train on text portion
```

Result:
- **Prompt**: -100 → ignored in loss
- **Neural tokens**: -100 → ignored in loss
- **Target text**: token IDs → full training signal
- Model ONLY learns to predict text from neural signals

### Session-Specific Read-in (Drift Mitigation)

Electrode drift (change in signal quality across days) is a major challenge. Solution:
- Per-session linear transformation: `x = Linear(session_id)(x)`
- Learns to normalize signal distribution per recording session
- Enables transfer across sessions without explicit calibration

### Attention Masking Strategy

Proper attention masking prevents the model from attending to:
1. **Padded neural positions**: Beyond the actual neural sequence length
2. **Padded text positions**: Beyond the actual text sequence length

```python
attention_mask = torch.ones((batch_size, seq_len), device=device)
# Mask padded neural positions
for i in range(batch_size):
    actual_end = start_len + patched_lengths[i]
    attention_mask[i, actual_end : start_len + projected_len] = 0
# Mask padded text positions
attention_mask[:, prefix_len:] = text_attention_mask
```

## 📊 Data Format Requirements

### HDF5 File Structure

Expected structure for `data_train.hdf5` and `data_val.hdf5`:

```
data_train.hdf5
├── trial_0001
│   ├── neural              (T, 512) - neural signal
│   ├── text                (1,) - target sentence
│   ├── text_ids            (L,) - tokenized text (optional)
│   ├── phonemes            (P,) - phoneme sequence (optional)
│   └── session (attr)      - session ID string
├── trial_0002
└── ...
```

### Session Statistics JSON

```json
{
  "session_001": {
    "mean": [0.1, 0.05, ...],  // 512-dim per-channel mean
    "std": [1.2, 0.9, ...]      // 512-dim per-channel std
  },
  "session_002": {...},
  ...
}
```

## 🚀 Running on JarvisLabs GPU Instances

The training script includes automatic instance pause on completion:

```bash
# Terminal 1: SSH into instance
ssh user@jarvislabs.instance.ip

# Terminal 2: Run training (will auto-pause on completion)
python scripts/train_e2e.py \
    --train_h5 /mnt/data/train/ \
    --val_h5 /mnt/data/val/ \
    --session_stats session_stats.json \
    --output_dir outputs/e2e_v2 \
    --epochs 100
```

The script will:
1. Train until completion or crash
2. Log everything to `outputs/e2e_v2/e2e_train.log`
3. Automatically send pause request to JarvisLabs API

## 📖 Technical Documentation

For deep dives into specific components, see:
- [COMPREHENSIVE_TECHNICAL_SPEC.md](COMPREHENSIVE_TECHNICAL_SPEC.md) - Full module registry, math, and dataflow
- [debug_e2e_training.md](debug_e2e_training.md) - Troubleshooting common errors
- [Zhang et al. (2025).pdf](Zhang%20et%20al.%20%282025%29.pdf) - Original paper (sections 3.2 and appendices A-R)

## 🔗 References

- **Paper**: Zhang et al. (2025). "A cross-species neural foundation model for end-to-end speech decoding"
- **Competition**: Kaggle Brain-to-Text '25 Competition (https://www.kaggle.com/competitions/brain-to-text-25/)
- **Framework**: PyTorch 2.0+, Transformers 4.45+, PEFT 0.7+
- **Base LLM**: Qwen2.5-1.5B-Instruct (Alibaba)
- **Quantization**: bitsandbytes (nf4)

## 📝 Citation

If you use this code or approach, please cite:

```bibtex
@misc{bit2025,
  title={BIT: Brain-to-Text Integration Transformer},
  author={Anonymous},
  year={2025},
  howpublished={Kaggle Brain-to-Text Competition},
  url={https://github.com/yourusername/brain2text2025}
}
```

## ⚖️ License

This project is provided for research and competition purposes. See individual files for specific licenses.

---

**Last Updated**: May 2026  
**Contributors**: [Your Name]  
**Status**: Active Development
