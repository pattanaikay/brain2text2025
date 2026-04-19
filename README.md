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

## Repository Structure

```text
.
├── approach #1- CNN + BiGRU/      # CNN-BiGRU implementation
│   ├── scripts/                   # Training and submission scripts
│   ├── src/                       # Model and dataloading source code
│   └── data/                      # Data utilities and visualizations
├── approach #2- CNN + BiLSTM + ngram/ # CNN-BiLSTM + n-gram implementation
│   ├── scripts/                   # Includes n-gram training and main pipeline
│   ├── src/                       # Source code including n-gram logic
│   └── data/                      # Data utilities
└── ...
```

## Getting Started

Each approach folder contains its own `requirements.txt` and specific instructions. Generally:

1. **Install Dependencies**:
   ```bash
   pip install torch h5py scipy numpy tqdm
   ```
2. **Data Setup**: Ensure the competition data is placed in the expected directory (usually `../t15_copyTask_neuralData/`).
3. **Training**: Navigate to an approach folder and run the training script:
   ```bash
   python scripts/train.py
   ```

## References
- **Competition**: [Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25)
- **Data Source**: [Dryad Dataset](https://doi.org/10.5061/dryad.dncjsxm85)
- **Lab**: [UC Davis Neuroprosthetics Lab](https://neuroprosthetics.science/)
