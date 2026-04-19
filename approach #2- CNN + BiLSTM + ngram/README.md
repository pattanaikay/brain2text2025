# Brain-to-Text 2025: Neural Speech Decoding Challenge

This repository contains a solution for the **[Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25)**, which aims to decode speech directly from intracortical neural activity.

## Overview

### Competition Challenge

The goal is to develop algorithms that decode attempted speech from brain activity recorded by 256 microelectrodes in speech motor cortex. This has critical applications for people with ALS (Amyotrophic Lateral Sclerosis) and brainstem stroke, who have lost the ability to move and speak.

- **Dataset**: 10,948 sentences from a single participant (T15) across 45 recording sessions spanning 20 months
- **Evaluation Metric**: Word Error Rate (WER) on 1,450 held-out test sentences
- **Baseline Performance**: 6.70% WER
- **Target**: Improve upon the baseline using novel architectures, data augmentation, language modeling, or other techniques

### Dataset Details

From the UC Davis Neuroprosthetics Lab, the dataset includes:

- **Recording Source**: 256 intracortical microelectrodes in speech motor cortex
- **Trial Data**: Neural spiking activity + sentence transcripts
- **Partitions**: Train / Validation / Test sets across multiple sessions
- **Sentence Corpuses**: Switchboard, 50-word vocabulary, Harvard sentences, random words, custom high-frequency words
- **Speaking Strategies**: Attempted vocalized (~30 wpm) and attempted silent (~50 wpm)
- **Data Format**: HDF5 files organized by session date

**Data Source**: https://doi.org/10.5061/dryad.dncjsxm85

## Project Structure

```
Brain2Text2025/
├── brain2text2025/
│   ├── data/                          # Data loading and analysis
│   │   ├── dataloading.py            # HDF5 file parsing
│   │   └── neuraldata_viz.py         # Visualization utilities
│   │
│   ├── src/                           # Core implementation
│   │   ├── dataloader.py             # PyTorch Dataset and collate functions
│   │   ├── models/
│   │   │   └── baseline.py           # CNN-BiLSTM architecture
│   │   └── preprocessing/
│   │       ├── dataloader.py         # Additional data utilities
│   │       └── utils.py              # Normalization, smoothing, stats
│   │
│   └── scripts/
│       └── train.py                  # Training pipeline with mixed precision
│
├── t15_copyTask_neuralData/           # Raw dataset (organized by session)
│   └── hdf5_data_final/
│       ├── t15.2023.08.11/
│       ├── t15.2023.08.13/
│       └── ... (45 total sessions through 2025)
│
└── t15_pretrained_rnn_baseline/       # Baseline model checkpoints
    └── checkpoint/
```

## Model Architecture

The solution implements a **CNN-BiLSTM hybrid architecture** for temporal sequence-to-sequence learning:

### Architecture Details

```
Input: (Batch, Time, 512 neural channels)
    ↓
CNN (Spatial-Temporal Feature Extraction)
  - Conv1d(512 → 512, kernel=3)
  - BatchNorm + ReLU + Dropout(0.2)
  - Conv1d(512 → 512, kernel=3)
  - BatchNorm + ReLU
    ↓
Bidirectional LSTM (Sequence Modeling)
  - 2 layers, 512 hidden units
  - Bidirectional context
  - Dropout(0.3)
    ↓
Output: (Batch, Time, num_classes)
    ↓
CTC Loss (Connectionist Temporal Classification)
```

### Key Features

- **CNN Layers**: Learn spatial-temporal patterns across 512 neural channels
- **BiLSTM**: Capture bidirectional temporal dependencies in neural sequences
- **CTC Loss**: Handle variable-length sequences without frame-level alignment
- **Mixed Precision Training**: Optimized for 6GB+ VRAM using AMP (Automatic Mixed Precision)

## Preprocessing Pipeline

### 1. Z-Score Normalization (Per-Session)

Neural signals are inherently non-stationary. Due to electrode impedance drift and physical array shifts, channel baselines change between recording sessions. We normalize per session to handle this:

$$Z = \frac{x - \mu}{\sigma}$$

- Computed across all trials in a session for each channel
- Handles dead channels by adding epsilon (1e-8) to prevent division by zero

### 2. Gaussian Smoothing

Neural firing is stochastic. Gaussian filtering the time series acts as a low-pass filter to reveal underlying motor intent:

- **Sigma**: 1.5 (covers ~30-40ms at 20ms bins)
- Applied per-channel along the time axis
- Reduces jitter while preserving critical neural dynamics

## Training

### Configuration

```python
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OPTIMIZER = AdamW (with weight decay: 1e-2)
LOSS = CTC Loss (blank index: 0)
DEVICE = CUDA (with AMP scaling)
```

### Mixed Precision Training

- **Purpose**: Reduce memory usage and accelerate computation
- **Implementation**: `torch.cuda.amp.autocast()` + GradScaler
- **Compatibility**: Optimized for GPUs with 6GB+ VRAM

### Data Loading

- **Custom Collate Function**: Handles variable-length neural sequences
- **Padding**: Sequences padded to max length in batch
- **Pin Memory**: Enabled for faster GPU transfer
- **Workers**: 4 parallel data loading processes

## Usage

### Installation

```bash
# Clone repository
git clone <repo_url>
cd Brain2Text2025

# Install dependencies
pip install torch torchvision torchaudio pytorch-cuda=12.1
pip install h5py scipy numpy
pip install tqdm tensorboard  # Optional: for monitoring
```

### Running Training

```python
python brain2text2025/scripts/train.py
```

### Making Predictions

1. Load trained model checkpoint
2. Process test sessions from `t15_copyTask_neuralData/hdf5_data_final/`
3. Generate phoneme predictions
4. Apply language model rescoring (n-gram + LLM)
5. Format output as CSV with columns: `[id, text]`

### Expected Output Format

Submission CSV file with chronological test trials:
```
id,text
0,the quick brown fox
1,jumps over the lazy dog
...
1449,...
```
## Dataset Description

Each HDF5 file (`data_train.hdf5`, `data_val.hdf5`, `data_test.hdf5`) contains trials with:

- `input_features`: Neural activity matrix (Time × 512 channels)
- `n_time_steps`: Trial duration
- `seq_class_ids`: Phoneme class IDs (if available)
- `seq_len`: Phoneme sequence length
- `transcription`: Ground-truth sentence text
- Metadata: session, block_num, trial_num


## References

### Primary Citation

Card, N., Wairagkar, M., Iacobacci, C., et al. (2025). "Brain-to-text '25." Kaggle Competition. https://kaggle.com/competitions/brain-to-text-25

### Related Work

- **Original NEJM Publication**: Card et al. "An Accurate and Rapidly Calibrating Speech Neuroprosthesis" (2024) *New England Journal of Medicine*
- **Previous Challenge**: [Brain-to-Text 2024](https://eval.ai/web/challenges/challenge-page/2099/evaluation) (baseline improved from 11.06% to 5.77% WER)
- **Research Lab**: [UC Davis Neuroprosthetics Lab](https://neuroprosthetics.science/) (part of [BrainGate Consortium](https://www.braingate.org/))

## Resources

- **Baseline Algorithm**: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- **Dataset (Dryad)**: https://doi.org/10.5061/dryad.dncjsxm85
- **Competition Discussion**: Kaggle discussion board
- **Contact**: nscard@health.ucdavis.edu

## License

The dataset is licensed under **CC0 1.0 Universal** (Public Domain Dedication). Please cite the original work in any publications.

---
