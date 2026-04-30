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

### 2. [Approach #2: CNN + BiLSTM + n-gram](./approach%20%232-%20CNN%20+%20BiLSTM%20+%20ngram/)
- **Architecture**: 1D Convolutional layers followed by a 2-layer Bidirectional LSTM.
- **Language Modeling**: Incorporates an n-gram (3-gram) language model for rescoring/decoding to improve transcript accuracy.
- **Features**: Similar preprocessing to Approach #1 with added linguistic priors.

### 3. [Approach #5: Transformer + Aero-1-Audio](./approach%20%235-%20Transformer%20+%20Aero-1-Audio/)
- **Architecture**: Transformer encoder (7 layers, 6 heads) for neural feature extraction, integrated with multimodal Aero-1-Audio LLM (1.5B parameters).
- **Fine-Tuning**: Uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning with 4-bit quantization.
- **Training**: Two-stage pipeline with optional self-supervised pretraining followed by supervised fine-tuning.
- **Features**: Leverages pretrained audio/speech understanding from multimodal LLM for improved accuracy.

## Repository Structure

```text
.
├── approach #1- CNN + BiGRU/           # CNN-BiGRU implementation
│   ├── scripts/                        # Training and submission scripts
│   ├── src/                            # Model and dataloading source code
│   └── data/                           # Data utilities and visualizations
├── approach #2- CNN + BiLSTM + ngram/  # CNN-BiLSTM + n-gram implementation
│   ├── scripts/                        # Includes n-gram training and main pipeline
│   ├── src/                            # Source code including n-gram logic
│   └── data/                           # Data utilities
├── approach #5- Transformer + Aero-1-Audio/  # Transformer + multimodal LLM
│   ├── scripts/                        # Training with SSL pretraining and supervised FT
│   ├── src/                            # Neural encoder, LLM integration, dataloading
│   └── requirements.txt                # Transformers, LoRA, BitsAndBytes dependencies
└── ...
```

## Getting Started

Each approach folder contains its own `requirements.txt` and specific instructions.

### Quick Start for Each Approach

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
python scripts/train.py --epochs 50 # Train with supervised fine-tuning
python scripts/submission.py         # Generate predictions
```

### General Setup

1. **Data Setup**: Ensure the competition data is placed in the expected directory (usually `../t15_copyTask_neuralData/`).
2. **GPU Requirements**: All approaches benefit from CUDA-capable GPU:
   - Approach #1-2: 6GB+ VRAM recommended
   - Approach #5: 16GB+ VRAM recommended (due to LLM quantization)
3. **Python Version**: Python 3.10+ recommended

## Approach Comparison

| Feature | Approach #1 | Approach #2 | Approach #5 |
|---------|------------|------------|------------|
| **Architecture** | CNN + BiGRU | CNN + BiLSTM | Transformer + LLM |
| **Decoding** | Greedy | Beam Search | LLM generation |
| **Language Model** | None | Explicit n-gram | Implicit (LLM) |
| **Memory (Training)** | ~6GB | ~6GB | ~16GB+ |
| **Training Time** | ~2-4 hrs | ~3-5 hrs | ~4-8 hrs |
| **Inference Speed** | Fast | Medium | Slow |
| **Expected WER** | Baseline | Better | Best |
| **Complexity** | Low | Medium | High |

## References
- **Competition**: [Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25)
- **Data Source**: [Dryad Dataset](https://doi.org/10.5061/dryad.dncjsxm85)
- **Aero-1-Audio**: [LMMS-Lab Multimodal LLM](https://github.com/LMM-Lab/Aero)
- **LoRA**: [Hu et al. 2021 - Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- **Lab**: [UC Davis Neuroprosthetics Lab](https://neuroprosthetics.science/)
