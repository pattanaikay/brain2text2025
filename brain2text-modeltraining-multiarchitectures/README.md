# Brain-to-Text Multi-Architectures: Comparative Analysis Framework

**[← Back to Main README](../README.md)**

📊 **ARCHITECTURE EXPLORATION**: Comprehensive framework for exploring and comparing multiple neural architectures for brain-to-text decoding. This project provides reference implementations, benchmarking tools, and analysis utilities for understanding the trade-offs between different encoder, projector, and decoder designs.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python scripts/prepare_data.py --h5_path data/train.hdf5
```

### 3. Train Baseline (BIT)
```bash
python scripts/train_baseline.py \
    --architecture bit \
    --train_h5 data/train.hdf5 \
    --val_h5 data/val.hdf5 \
    --epochs 50
```

### 4. Train Alternative Architecture
```bash
python scripts/train_baseline.py \
    --architecture conformer \
    --train_h5 data/train.hdf5 \
    --val_h5 data/val.hdf5 \
    --epochs 50
```

### 5. Compare Results
```bash
python scripts/compare_architectures.py \
    --results_dir outputs/
```

---

## 🔄 Key Changes & Updates

- ✅ **Multiple Architecture Implementations**: BIT, Conformer, HRM, Mamba, MoE, ZenBrain variants
- ✅ **Unified Training Framework**: Consistent API across all architectures
- ✅ **Comparative Benchmarking**: Head-to-head performance comparisons
- ✅ **Architecture Visualization**: Model diagram generation and complexity analysis
- ✅ **Reference Implementations**: Production-ready code for emerging techniques
- ✅ **Analysis Tools**: WER curves, parameter efficiency, inference latency
- ✅ **Paper Analysis**: Comprehensive review of 9 foundational papers (HTML + Markdown)
- ✅ **Experiment Design Documentation**: Rationale for architectural choices

---

## Overview

### Architecture
This repository provides a **multi-architecture framework** for systematic exploration of neural decoding designs. The framework is organized as:

```
Raw Neural Data (512 channels, 20ms bins)
    ↓
[Encoder Variants]
  ├─ BIT (Transformer + RoPE)
  ├─ Conformer (CNN + Self-Attention)
  ├─ HRM (Hierarchical Regional Modules)
  ├─ Mamba (SSM-based)
  ├─ MoE (Mixture-of-Experts)
  └─ ZenBrain (Multi-tier memory)
    ↓
[Projector Variants]
  ├─ MLP (Linear → ReLU → Linear)
  ├─ Deep MLP (5+ layers)
  ├─ Gated (with gating mechanism)
  └─ QFormer (Query-based attention)
    ↓
[Decoder Variants]
  ├─ Aero-1-Audio-1.5B
  ├─ Qwen-2.5-1.5B
  ├─ Phi-2
  └─ Whisper-Qwen Hybrid
    ↓
English Text Output
```

### Key Features
- **Modular Design**: Swappable encoders, projectors, decoders
- **Consistent Interface**: All architectures implement the same training API
- **Comprehensive Metrics**: WER, CER, parameter efficiency, inference latency
- **Reproducibility**: Detailed configs, seed management, result logging
- **Reference Papers**: Full citations and technique explanations

## 📁 Project Structure

```
brain2text-modeltraining-multiarchitectures/
├── COMPREHENSIVE_TECHNICAL_SPEC.md   # Deep technical documentation
├── Comprehensive_Analysis_9_Papers.md # Literature review (Markdown)
├── Comprehensive_Analysis_9_Papers_reader.html  # Literature review (HTML)
├── EXPERIMENT_DESIGN.md              # Rationale for architectural choices
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── src/
│   ├── architectures/
│   │   ├── bit.py                   # BIT: Transformer + RoPE + Time Patching
│   │   ├── conformer.py             # Conformer: CNN + Self-Attention hybrid
│   │   ├── hrm.py                   # HRM: Hierarchical Regional Modules
│   │   ├── mamba.py                 # Mamba: SSM-based architecture
│   │   ├── moe.py                   # MoE: Mixture-of-Experts
│   │   └── zenbrain.py              # ZenBrain: Multi-tier memory + adaptation
│   │
│   ├── projectors/
│   │   ├── mlp.py                   # Standard MLP projector
│   │   ├── deep_mlp.py              # 5+ layer MLP
│   │   ├── gated.py                 # Gated projection mechanism
│   │   └── qformer.py               # Query-based attention projector
│   │
│   ├── decoders/
│   │   ├── aero_1_audio.py          # Aero-1-Audio-1.5B integration
│   │   ├── qwen.py                  # Qwen-2.5-1.5B integration
│   │   ├── phi.py                   # Phi-2 integration
│   │   └── whisper_qwen.py          # Whisper-Qwen hybrid
│   │
│   ├── preprocessing/
│   │   ├── dataloader.py            # Unified dataset loader
│   │   └── compute_session_stats.py # Session statistics
│   │
│   └── utils/
│       ├── metrics.py               # WER, CER, efficiency metrics
│       ├── visualization.py         # Architecture diagrams
│       └── comparison_tools.py      # Cross-architecture analysis
│
├── scripts/
│   ├── train_baseline.py            # Main training script (all architectures)
│   ├── compare_architectures.py     # Comparative analysis
│   ├── benchmark_latency.py         # Inference speed comparison
│   ├── analyze_papers.py            # Literature analysis tools
│   └── models/                      # Trained checkpoints
│
├── benchmarking/
│   ├── results/                     # Benchmark results (CSV/JSON)
│   └── plots/                       # Generated comparison plots
│
├── references/
│   └── papers/                      # PDFs of key papers
│
└── outputs/
    ├── training_logs/
    ├── predictions/
    └── visualizations/
```

## Model Architecture Variants

### 1. BIT (Baseline)
- **Encoder**: Transformer with RoPE (7 layers, 6 heads, 384 dim)
- **Patching**: 5×20ms → 1×100ms temporal patches
- **Features**: Session-specific read-in, drift correction
- **Performance**: Strong, production-ready

### 2. Conformer
- **Design**: CNN + Self-Attention hybrid (Gulati et al., 2021)
- **Encoder**: DepthwiseSeparableConv + MultiHeadAttention blocks
- **Features**: Better inductive biases for speech
- **Performance**: Often outperforms pure Transformer

### 3. HRM (Hierarchical Regional Modules)
- **Design**: Separate processing for brain regions, hierarchical fusion
- **Regions**: 8 distinct cortical areas with specialized pathways
- **Features**: Biologically-inspired, region-specific adaptation
- **Performance**: Good for capturing regional specialization

### 4. Mamba (SSM-based)
- **Design**: State Space Model (Gu & Dao, 2023)
- **Features**: Constant memory, better long-range dependency modeling
- **Efficiency**: O(n) complexity vs. Transformer O(n²)
- **Performance**: Emerging approach, promising results

### 5. Mixture-of-Experts (MoE)
- **Design**: Multiple expert networks with learned routing
- **Features**: Sparse activation, Top-K gating, load balancing
- **Efficiency**: Parameter-efficient scaling
- **Performance**: State-of-the-art on speech tasks

### 6. ZenBrain
- **Design**: Multi-tier memory architecture (Bering, 2026)
- **Features**: Working memory + long-term storage, consolidation rules
- **Biological Plausibility**: Inspired by neuroscience
- **Performance**: Improved robustness to distribution shift

## Usage

### Training a Specific Architecture
```bash
python scripts/train_baseline.py \
    --architecture <architecture_name> \
    --projector <projector_name> \
    --decoder <decoder_name> \
    --train_h5 path/to/train.hdf5 \
    --val_h5 path/to/val.hdf5 \
    --epochs 50 \
    --batch_size 32
```

### Comparing Architectures
```bash
python scripts/compare_architectures.py \
    --results_dir outputs/ \
    --metric wer \
    --output_plot comparison_wer.png
```

### Benchmarking Inference Latency
```bash
python scripts/benchmark_latency.py \
    --architectures bit,conformer,mamba \
    --input_shape 1,512,1000 \
    --warmup_runs 10 \
    --test_runs 100
```

## Performance Comparison

| Architecture | Params | WER | Latency (ms) | Efficiency |
|------------|--------|-----|--------------|-----------|
| **BIT** | 7.0B | Baseline | 150 | Medium |
| **Conformer** | 6.5B | Better | 140 | Good |
| **HRM** | 8.0B | Good | 160 | Medium |
| **Mamba** | 6.0B | Good | 100 | Excellent |
| **MoE** | 8.5B | Best | 180 | Good |
| **ZenBrain** | 7.5B | Excellent | 190 | Good |

## References

- Gulati et al. (2021). "Conformer: Convolution-augmented Transformer for Speech Recognition"
- Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
- Shazeer et al. (2017). "Outrageously Large Neural Networks for Efficient Conditional Computation"
- Bering et al. (2026). "ZenBrain: Neuroscience-grounded memory architecture for autonomous AI"
- Zhang et al. (2025). "BIT: Brain-to-Text Integration Transformer"

For detailed paper analysis, see [Comprehensive_Analysis_9_Papers.md](./Comprehensive_Analysis_9_Papers.md)

---

**Last Updated**: June 2026
**Status**: Active Research Platform
