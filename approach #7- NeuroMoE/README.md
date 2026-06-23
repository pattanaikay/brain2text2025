# Approach #7: NeuroMoE - Mixture of Experts Neural Decoder

**[← Back to Main README](../README.md)**

This folder implements **NeuroMoE**, an advanced **Mixture-of-Experts (MoE) architecture** for state-of-the-art neural speech decoding. It combines region-specific expert modules with intelligent routing to achieve superior performance on the Kaggle Brain-to-Text 2025 Competition.

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Compute Session Statistics
```bash
python src/preprocessing/compute_session_stats.py --h5_list data/h5_list.json
```

### 3. Phase 1: SSL Pretraining with MoE
```bash
python scripts/train_ssl.py \
    --train_h5 path/to/train.hdf5 \
    --val_h5 path/to/val.hdf5 \
    --epochs 50 \
    --moe_type regional
```

### 4. Phase 2: End-to-End Fine-tuning
```bash
python scripts/train_e2e.py \
    --train_h5 path/to/train.hdf5 \
    --val_h5 path/to/val.hdf5 \
    --ssl_checkpoint scripts/models/ssl/best_encoder_ssl.pth \
    --session_stats session_stats.json \
    --moe_type regional
```

### 5. Evaluate
```bash
python scripts/evaluate.py \
    --test_h5 path/to/test.hdf5 \
    --checkpoint scripts/models/e2e/best_model_wer.pth \
    --analyze_routing true
```

---

## 📚 Overview

This repository implements **NeuroMoE**, a cutting-edge **Mixture-of-Experts (MoE)** architecture designed to achieve state-of-the-art performance in neural speech decoding. The approach combines:

1. **Sparse Mixture-of-Experts (SSMoE)**: Multiple specialized neural experts with learned routing
2. **Region-Specific Experts**: 8 brain-region processing modules (Broca, Wernicke, motor cortex, etc.)
3. **Intelligent Routing**: Top-K gating with load balancing to prevent expert collapse
4. **Hierarchical Architecture**: Shared base + region-specific branches + LLM integration

### Key Innovation: Neuroscience-Grounded Architecture
Unlike generic MoE models, NeuroMoE is designed with **brain anatomy in mind**:
- Separate expert pathways for different brain regions
- Region-specific receptive field processing
- Mimics functional specialization in motor cortex

### Key Features
- **8 Regional Experts**: Specialized modules for distinct brain areas (64 channels each)
- **Top-K Routing (K=2)**: Sparse expert activation with dynamic load balancing
- **Auxiliary Loss**: Prevents expert collapse and ensures balanced utilization
- **Session-Specific Read-in**: Per-session linear layers for drift correction
- **Multimodal Alignment**: Contrastive loss for neural-text space alignment
- **Time Patching**: Groups 20ms bins into 100ms patches
- **4-bit Quantized LLM**: Parameter-efficient fine-tuning of Aero-1-Audio-1.5B

---

## 🔄 Key Changes & Updates

- ✅ Advanced Mixture-of-Experts architecture with regional decomposition
- ✅ Top-K gating mechanism with load balancing
- ✅ 8 specialized expert modules for brain region processing
- ✅ Enhanced routing analysis and visualization tools
- ✅ Auxiliary loss implementation to prevent expert collapse
- ✅ Comprehensive routing statistics and expert utilization metrics
- ✅ Integration with EEGMoE (Gao et al., 2026) and BrainStack (Zhao et al., 2026) concepts
- ✅ Complete technical specification with MoE theory and implementation details

---

## Project Structure

```
approach #7- NeuroMoE/
├── scripts/
│   ├── train_ssl.py                  # Phase 1: SSL pretraining with MoE
│   ├── train_e2e.py                  # Phase 2: End-to-end fine-tuning
│   ├── evaluate.py                   # Generate predictions & analyze routing
│   ├── plot_metrics.py               # Visualize training + expert usage
│   ├── analyze_routing.py            # Deep analysis of routing patterns
│   ├── visualize_moe_architecture.py # Architecture diagrams
│   └── models/                       # Checkpoints
│
├── src/                               # Core implementation
│   ├── models/
│   │   ├── encoder.py                # Base encoder + regional experts
│   │   ├── moe_router.py             # Top-K gating and routing logic
│   │   ├── regional_experts.py       # 8 brain-region expert modules
│   │   ├── projector.py              # MLP modality projector
│   │   └── neuromoe.py               # NeuroMoE: Complete end-to-end model
│   ├── preprocessing/
│   │   ├── dataloader.py             # Preprocessed BCI dataset
│   │   ├── compute_session_stats.py  # Session statistics
│   │   └── brain_region_mapper.py    # Map channels to brain regions
│   └── utils/
│       ├── metrics.py                # WER, CER, expert balance metrics
│       ├── routing_analysis.py       # Analyze routing patterns
│       └── visualizations.py         # Routing heatmaps, expert usage graphs
│
├── requirements.txt                  # Python dependencies
├── COMPREHENSIVE_TECHNICAL_SPEC.md   # Detailed MoE theory & implementation
├── NeuroMoE_Changes.md               # Changelog and improvements
└── README.md                         # This file
```

## Model Architecture

### 1. Regional Brain Decomposition

NeuroMoE partitions the 512 neural channels into 8 specialized regions:

```
512 Channels (256 electrodes × 2 channels)
    ↓
[8 Regional Experts]
  ├─ Broca's Area (Speech production) - 64 channels
  ├─ Wernicke's Area (Speech comprehension) - 64 channels
  ├─ Primary Motor Cortex (Movement) - 64 channels
  ├─ Supplementary Motor Area (Planning) - 64 channels
  ├─ Premotor Cortex (High-level motor) - 64 channels
  ├─ Anterior Insula (Articulation) - 64 channels
  ├─ Inferior Parietal Lobule (Integration) - 64 channels
  └─ Temporal Pole (Semantic) - 64 channels
    ↓
[Top-K Router (K=2)]
  → Selects 2 best experts per token
    ↓
[Expert Outputs]
  → Weighted sum of top-K expert outputs
    ↓
Fused Representation (384-dim)
```

### 2. Mixture-of-Experts Layer

```
Input: (Batch, Time, 384-dim)
    ↓
[Gating Network]
  → Learns routing weights: logits = Linear(input)
    ↓
[Top-K Selection]
  → Select K=2 experts with highest gate values
  → Compute sparse weight matrix
    ↓
[Expert Computation]
  → Each expert: FFN(input) → 256-dim output
  → 8 experts in parallel
    ↓
[Sparse Weighted Sum]
  → output = sum(gate[k] * expert[k](input)) for k in top_K
    ↓
[Auxiliary Load Balancing Loss]
  → Encourages balanced expert utilization
  → Prevents expert collapse
```

### 3. Auxiliary Loss for Load Balancing

$$\mathcal{L}_{load\_balance} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i$ = fraction of tokens routed to expert $i$
- $P_i$ = average gating probability for expert $i$
- $\alpha$ = load balance coefficient

### 4. Complete Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_{cont} \mathcal{L}_{contrastive} + \lambda_{lb} \mathcal{L}_{load\_balance}$$

### 5. LLM Decoder
- **Base Model**: Qwen-2.5-1.5B (Audio-tuned)
- **Quantization**: 4-bit NF4 with double quantization
- **Adaptation**: LoRA on attention + projection layers

## Usage

### 1. Preparation
Compute session statistics:
```bash
python src/preprocessing/compute_session_stats.py --h5_list data/h5_list.json
```

### 2. Phase 1: SSL Pretraining with MoE
Learn robust representations with expert routing:
```bash
python scripts/train_ssl.py \
    --train_h5 path/to/train.hdf5 \
    --val_h5 path/to/val.hdf5 \
    --epochs 50 \
    --moe_type regional \
    --num_experts 8 \
    --top_k 2 \
    --load_balance_weight 0.01
```

### 3. Phase 2: End-to-End Fine-tuning
Fine-tune the full NeuroMoE pipeline:
```bash
python scripts/train_e2e.py \
    --train_h5 path/to/train.hdf5 \
    --val_h5 path/to/val.hdf5 \
    --ssl_checkpoint scripts/models/ssl/best_encoder_ssl.pth \
    --session_stats session_stats.json \
    --moe_type regional \
    --top_k 2 \
    --load_balance_weight 0.01
```

### 4. Analyze Expert Routing (Optional)
Deep-dive into routing patterns:
```bash
python scripts/analyze_routing.py \
    --checkpoint scripts/models/e2e/best_model_wer.pth \
    --test_h5 path/to/test.hdf5
```

### 5. Evaluation & Visualization
Generate predictions and routing analysis:
```bash
python scripts/evaluate.py \
    --test_h5 path/to/test.hdf5 \
    --checkpoint scripts/models/e2e/best_model_wer.pth \
    --analyze_routing true
python scripts/plot_metrics.py --history outputs/training_history.json
python scripts/visualize_moe_architecture.py
```

## Routing Analysis & Interpretation

NeuroMoE provides tools to understand expert utilization:

### Expert Load Metrics
- **Expert Activation Rate**: Fraction of tokens routed to each expert
- **Expert Load Balance**: Entropy of routing distribution (0=imbalanced, 1=uniform)
- **Regional Specialization**: Which regions are used for different inputs

### Visualizations
- **Routing Heatmaps**: Expert selection patterns over time
- **Expert Utilization Graph**: Bar chart of each expert's activation rate
- **Load Balance Curves**: Training dynamics of load balancing loss

## Comparison with Other Approaches

| Feature | #5 BIT | #6 BIT+Di | #7 NeuroMoE |
|---------|--------|-----------|-----------|
| **Architecture** | Transformer | Transformer | Transformer + MoE |
| **Expert Count** | 1 (Monolithic) | 1 | 8 Regional |
| **Routing** | N/A | N/A | Top-K (K=2) |
| **Parameters** | ~7B (Qwen) | ~7B | ~8.5B (with experts) |
| **Sparse Activation** | No | No | Yes (Top-K) |
| **Load Balancing** | N/A | N/A | Yes (Auxiliary Loss) |
| **Complexity** | High | High+ | Very High |
| **Expected WER** | Excellent | Excellent+ | **SOTA** |

## References

- Gao et al. (2026). "EEGMoE: Mixture-of-Experts for EEG Analysis"
- Zhao et al. (2026). "BrainStack: Regional Brain Processing Architecture"
- Bering et al. (2026). "ZenBrain: Neuroscience-grounded memory architecture for autonomous AI" (arXiv:2604.23878)
- Shazeer et al. (2017). "Outrageously Large Neural Networks for Efficient Conditional Computation" (GLaM)
- Lepikhin et al. (2020). "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"

For detailed technical information, see [COMPREHENSIVE_TECHNICAL_SPEC.md](./COMPREHENSIVE_TECHNICAL_SPEC.md).
