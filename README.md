# Brain-to-Text 2025: Neural Speech Decoding

This repository contains multiple approaches for the **[Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25)**. The goal of the competition is to decode attempted speech directly from intracortical neural activity recorded from the speech motor cortex.

## Project Overview

The project focuses on developing sequence-to-sequence models to translate neural spiking activity (512 features from 256 microelectrodes) into text. The dataset involves 10,948 sentences from a single participant (T15) across 45 recording sessions.

## Approaches

This repository explores different architectures and post-processing techniques to improve Word Error Rate (WER):

### 1. [Approach #1: CNN + BiGRU](./approach%20%231-%20CNN%20+%20BiGRU/)
- **Architecture**: 1D Convolutional layers for spatial-temporal feature extraction followed by a 2-layer Bidirectional GRU.
- **Loss**: Connectionist Temporal Classification (CTC) loss.
- **Features**: Per-session Z-score normalization and Gaussian smoothing.
- **Status**: Baseline model for initial exploration.

### 2. [Approach #2: CNN + BiLSTM + n-gram](./approach%20%232-%20CNN%20+%20BiLSTM%20+%20ngram/)
- **Architecture**: 1D Convolutional layers followed by a 2-layer Bidirectional LSTM.
- **Language Modeling**: Incorporates an n-gram (3-gram) language model for rescoring/decoding to improve transcript accuracy.
- **Decoding**: Beam search with language model scoring.
- **Features**: Similar preprocessing to Approach #1 with added linguistic priors.
- **Status**: Improved baseline with linguistic constraints.

### 3. [Approach #5: Transformer + Aero-1-Audio](./approach%20%235-%20Transformer%20+%20Aero-1-Audio/)
- **Architecture**: Transformer encoder (7 layers, 6 heads, 384 dim) with Rotary Position Embeddings (RoPE) for neural feature extraction, integrated with multimodal Aero-1-Audio LLM (1.5B parameters).
- **Time Patching**: Groups 20ms neural bins into 100ms patches (5 bins) to improve long-range attention and reduce sequence length.
- **Drift Correction**: Session-specific read-in layers to handle probe drift across recording sessions.
- **Training**: Three-phase pipeline:
  - Phase 1: Self-Supervised Learning (Masked Neural Modeling)
  - Phase 2: End-to-End Fine-tuning with Contrastive Loss
  - Phase 3: Optional CTC-based Phoneme Recognition
- **Fine-Tuning**: Uses QLoRA (4-bit quantization) for parameter-efficient adaptation of Aero-1-Audio-1.5B.
- **Loss Function**: Combined cross-entropy + InfoNCE contrastive loss ($\mathcal{L}_{BIT} = \mathcal{L}_{CE} + \mathcal{L}_{contrastive}$).
- **Status**: Advanced multimodal approach with strong performance.

### 4. [Approach #6: Transformer + Aero-1-Audio + Diphones](./approach%20%236-%20Transformer%20+%20Aero-1-Audio%20+%20Diphones/)
- **Architecture**: Similar to Approach #5 with extended support for diphone-level linguistic constraints.
- **Linguistic Enhancement**: Incorporates diphone (two-phoneme) sequences for more granular phonetic modeling.
- **Features**: Builds on Approach #5's time patching and drift correction with enhanced linguistic priors.
- **Status**: Experimental variant exploring phonetic granularity.

## Repository Structure

```text
.
├── approach #1- CNN + BiGRU/                    # CNN-BiGRU baseline implementation
│   ├── scripts/                                 # Training and submission scripts
│   ├── src/                                     # Model and dataloading source code
│   └── data/                                    # Data utilities and visualizations
│
├── approach #2- CNN + BiLSTM + ngram/           # CNN-BiLSTM with n-gram LM
│   ├── scripts/                                 # Includes n-gram training and main pipeline
│   ├── src/                                     # Source code including n-gram logic
│   └── data/                                    # Data utilities
│
├── approach #5- Transformer + Aero-1-Audio/    # Transformer + multimodal LLM (BIT v1)
│   ├── scripts/                                 # SSL pretraining, supervised FT, CTC training
│   ├── src/                                     # Neural encoder, LLM integration, dataloading
│   └── requirements.txt                         # Transformers, LoRA, BitsAndBytes dependencies
│
├── approach #6- Transformer + Aero-1-Audio + Diphones/  # BIT with diphone support (BIT v2)
│   ├── scripts/                                 # Training and evaluation scripts
│   ├── src/                                     # Enhanced encoder with diphone modeling
│   └── requirements.txt                         # Project dependencies
│
├── brain2text-modeltraining/                    # **MAIN: Production BIT Framework Implementation**
│   ├── scripts/                                 # Training pipeline (SSL, E2E, CTC)
│   │   ├── train_ssl.py                        # Phase 1: Self-supervised pretraining
│   │   ├── train_e2e.py                        # Phase 2: End-to-end fine-tuning
│   │   ├── train_ctc.py                        # Phase 3: CTC phoneme recognition
│   │   ├── evaluate.py                         # Evaluation and metric calculation
│   │   ├── plot_metrics.py                     # Visualization of training metrics
│   │   └── models/                             # Checkpoints and saved models
│   ├── src/                                     # Core implementation
│   │   ├── models/                             # BIT_Transformer, projectors, baseline
│   │   ├── preprocessing/                      # Dataloaders and preprocessing utilities
│   │   └── utils/                              # Metrics, helpers
│   ├── outputs/                                 # Training outputs (logs, histories)
│   ├── data/                                    # Data utilities
│   ├── requirements.txt                         # Python dependencies
│   └── COMPREHENSIVE_TECHNICAL_SPEC.md         # Detailed technical documentation
│
├── bci-incremental-roadmap.md                   # Development roadmap and architecture evolution
├── index.html                                   # Project overview HTML
└── README.md                                    # This file
```

## Getting Started

Each approach folder contains its own `requirements.txt` and specific instructions. The **`brain2text-modeltraining/`** folder is the primary production implementation of the BIT framework.

### Quick Start for Main Implementation (BIT Framework)

```bash
cd brain2text-modeltraining/
pip install -r requirements.txt

# Step 1: Compute session statistics for normalization
python src/preprocessing/compute_session_stats.py --h5_list data/h5_list.json

# Step 2: Phase 1 - Self-Supervised Pretraining (optional)
python scripts/train_ssl.py --epochs 50

# Step 3: Phase 2 - End-to-End Fine-tuning
python scripts/train_e2e.py --checkpoint scripts/models/ssl/best_encoder_ssl.pth --epochs 100

# Step 4: Phase 3 - CTC Phoneme Recognition (optional)
python scripts/train_ctc.py --checkpoint scripts/models/e2e/best_model_wer.pth --epochs 50

# Step 5: Evaluate and Generate Predictions
python scripts/evaluate.py --checkpoint scripts/models/e2e/best_model_wer.pth
python scripts/plot_metrics.py
```

### Quick Start for Alternative Approaches

**Approach #1 (CNN + BiGRU)**:
```bash
cd approach\ #1-\ CNN\ +\ BiGRU/
pip install -r requirements.txt
python scripts/train.py
```

**Approach #2 (CNN + BiLSTM + n-gram)**:
```bash
cd approach\ #2-\ CNN\ +\ BiLSTM\ +\ ngram/
pip install -r requirements.txt
python scripts/train.py              # Train neural model
python scripts/train_ngram.py        # Train n-gram LM
python scripts/submission.py         # Generate predictions
```

**Approach #5 (Transformer + Aero-1-Audio)**:
```bash
cd approach\ #5-\ Transformer\ +\ Aero-1-Audio/
pip install -r requirements.txt
python scripts/train_ssl.py --epochs 50          # Optional pretraining
python scripts/train_e2e.py --epochs 100         # Fine-tuning
python scripts/evaluate.py                       # Evaluation
```

**Approach #6 (Transformer + Aero-1-Audio + Diphones)**:
```bash
cd approach\ #6-\ Transformer\ +\ Aero-1-Audio\ +\ Diphones/
pip install -r requirements.txt
python scripts/train_ssl.py --epochs 50          # Optional pretraining
python scripts/train_e2e.py --epochs 100         # Fine-tuning
python scripts/evaluate.py                       # Evaluation
```

### General Setup

1. **Data Setup**: Ensure the competition data is placed in the expected directory (usually `../t15_copyTask_neuralData/`).
2. **GPU Requirements**: 
   - Approach #1-2: 6GB+ VRAM recommended
   - Approach #5-6 & brain2text-modeltraining: 16GB+ VRAM recommended (due to LLM quantization and transformer computations)
3. **Python Version**: Python 3.10+ recommended
4. **Recommended Approach**: Use `brain2text-modeltraining/` for the most up-to-date implementation with all advanced features.

### Project Documentation

- **[BCI Incremental Roadmap](./bci-incremental-roadmap.md)**: Documents the iterative development from baseline RNN to the BIT framework
- **[Main Technical Specification](./brain2text-modeltraining/COMPREHENSIVE_TECHNICAL_SPEC.md)**: Comprehensive technical details of the BIT framework
- **[Approach #5 Specification](./approach%20%235-%20Transformer%20+%20Aero-1-Audio/COMPREHENSIVE_TECHNICAL_SPEC.md)**: Detailed architecture documentation for Approach #5
- **[Approach #6 Specification](./approach%20%236-%20Transformer%20+%20Aero-1-Audio%20+%20Diphones/COMPREHENSIVE_TECHNICAL_SPEC.md)**: Detailed documentation for the diphone variant

## Approach Comparison

| Feature | Approach #1 | Approach #2 | Approach #5 | Approach #6 |
|---------|------------|------------|------------|------------|
| **Architecture** | CNN + BiGRU | CNN + BiLSTM | Transformer + LLM | Transformer + LLM + Diphones |
| **Decoding** | Greedy | Beam Search | LLM generation | LLM generation |
| **Language Model** | None | Explicit n-gram | Implicit (LLM) | Implicit (LLM) + Diphone LM |
| **Position Encoding** | N/A | N/A | RoPE | RoPE |
| **Drift Correction** | Session norm | Session norm | Session read-in | Session read-in |
| **Memory (Training)** | ~6GB | ~6GB | ~16GB+ | ~16GB+ |
| **Training Time** | ~2-4 hrs | ~3-5 hrs | ~6-10 hrs | ~6-10 hrs |
| **Inference Speed** | Fast | Medium | Slow | Slow |
| **Expected WER** | Baseline | Better | Best | Excellent |
| **Complexity** | Low | Medium | High | High+ |
| **Status** | Baseline | Improved | Production | Experimental |

## References
- **Competition**: [Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25)
- **Data Source**: [Dryad Dataset](https://doi.org/10.5061/dryad.dncjsxm85)
- **Aero-1-Audio**: [LMMS-Lab Multimodal LLM](https://github.com/LMM-Lab/Aero)
- **LoRA**: [Hu et al. 2021 - Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **Lab**: [UC Davis Neuroprosthetics Lab](https://neuroprosthetics.science/)
