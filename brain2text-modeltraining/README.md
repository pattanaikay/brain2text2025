# Brain-to-Text 2025: Approach #5 - BIT Framework (Transformer + Aero-1-Audio)

This repository implements the **BIT (Brain-to-Text Integration Transformer)** framework for the Kaggle Brain-to-Text 2025 Competition. This approach leverages a modern multimodal LLM (Aero-1-Audio) integrated with a Transformer-based neural encoder, optimized for high-performance neural decoding.

## Overview

### Key Features
- **Time Patching**: Groups 20ms neural bins into 100ms patches (5 bins) to reduce sequence length and improve long-range attention.
- **Subject-Specific Read-in**: Session-specific linear layers to handle probe drift across different recording days.
- **Multimodal Alignment**: Uses InfoNCE Contrastive Loss to align neural and text representations in a shared latent space.
- **End-to-End Fine-tuning**: Joint optimization of CE loss and Contrastive loss ($\mathcal{L}_{BIT} = \mathcal{L}_{CE} + \mathcal{L}_{contrastive}$).
- **Parameter-Efficient Tuning**: 4-bit QLoRA adaptation of Aero-1-Audio-1.5B.

## Project Structure

```
approach #5- Transformer + Aero-1-Audio/
├── scripts/
│   ├── train_ssl.py                  # Phase 1: Masked Neural Modeling (Pretraining)
│   ├── train_e2e.py                  # Phase 2: End-to-End BIT Fine-tuning
│   ├── train_ctc.py                  # Phase 3: CTC-Loss Phoneme Recognition
│   ├── evaluate.py                   # Generate test predictions and calculate metrics (WER/CER)
│   ├── plot_metrics.py               # Visualize training dynamics (Loss, WER, CER)
│   ├── visualize_architecture.py     # Generate architecture visualizations using TorchView
│   ├── dry_run.py                    # Preflight validation of model initialization and forward pass
│   └── models/                       # Checkpoints
│
├── src/                               # Core implementation
│   ├── models/
│   │   ├── encoder.py                # BIT_Transformer with Time Patching
│   │   ├── projector.py              # 3-layer MLP Modality Projector
│   │   └── baseline.py               # BITModel (LLM Integration + Contrastive Loss)
│   ├── preprocessing/
│   │   ├── dataloader.py             # Preprocessed_BCI_Dataset with Gaussian Smoothing
│   │   └── compute_session_stats.py  # Utility for Z-score normalization
│   └── utils/
│       └── metrics.py                # WER and CER evaluation metrics
│
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Architecture

### 1. Neural Encoder (`BIT_Transformer`)
- **Type**: Transformer with Rotary Position Embeddings (RoPE) (7 layers, 6 heads, 384 dim)
- **Positional Encoding**: RoPE applied at each transformer block instead of fixed PE
- **Patching**: 5x20ms bins $\rightarrow$ 1x100ms patch via LayerNorm-sandwich embedding
- **Drift Correction**: Session-specific read-in layers using `nn.ModuleDict` for handling probe drift
- **Preprocessing**: Z-score normalization per session + Gaussian smoothing (σ=1.5)

### 2. Modality Projector
- **Architecture**: 5-layer MLP (`Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear`)
- **Hidden Dimension**: 1024
- **Mapping**: Projects 384-dim neural tokens to 1536-dim LLM latent space
- **Loss Function**: ModalityAlignmentLoss with learnable temperature parameter ($\tau$) for InfoNCE contrastive alignment

### 3. LLM Decoder (`Aero-1-Audio-1.5B`)
- **Base Model**: Qwen-2.5-1.5B (Audio-tuned)
- **Quantization**: 4-bit NF4 with double quantization
- **Adaptation**: LoRA on `q_proj, k_proj, v_proj, o_proj` and `audio_projector`

## Usage

### 0. Preflight Validation
Validate model initialization and forward pass (optional):
```bash
python scripts/dry_run.py
```

### 1. Preparation
Compute session statistics for Z-score normalization:
```bash
python src/preprocessing/compute_session_stats.py --h5_list data/h5_list.json
```

### 2. Phase 1: SSL Pretraining
Learn robust neural representations via masked patch modeling:
```bash
python scripts/train_ssl.py --train_h5 path/to/train.hdf5 --val_h5 path/to/val.hdf5 --epochs 50
```

### 3. Phase 2: End-to-End Fine-tuning
Fine-tune the full BIT pipeline with joint CE + Contrastive loss:
```bash
python scripts/train_e2e.py --train_h5 path/to/train.hdf5 --val_h5 path/to/val.hdf5 --ssl_checkpoint scripts/models/ssl/best_encoder_ssl.pth --session_stats session_stats.json
```

### 4. Phase 3: CTC Phoneme Recognition (Alternative)
Train with CTC loss for phoneme-level decoding:
```bash
python scripts/train_ctc.py --train_h5 path/to/train.hdf5 --val_h5 path/to/val.hdf5 --ssl_checkpoint scripts/models/ssl/best_encoder_ssl.pth
```

### 5. Evaluation & Visualization
Generate predictions and visualizations:
```bash
python scripts/evaluate.py --test_h5 path/to/test.hdf5 --checkpoint scripts/models/e2e/best_model_wer.pth
python scripts/plot_metrics.py --history outputs/training_history.json
python scripts/visualize_architecture.py  # Generate architecture diagrams
```

## References
- Zhang et al. (2025). "BIT: Brain-to-Text Integration Transformer."
- Cardio et al. (2025). "Brain-to-Text '25 Kaggle Competition."
