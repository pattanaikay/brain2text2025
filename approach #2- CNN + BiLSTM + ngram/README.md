# Brain-to-Text 2025: Approach #2 - CNN + BiLSTM + N-gram Language Model

This repository contains Approach #2 for the **[Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25)**, which aims to decode speech directly from intracortical neural activity. This approach enhances the CNN-BiLSTM architecture with n-gram language model integration for improved decoding accuracy.

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
approach #2- CNN + BiLSTM + ngram/
├── data/                              # Data loading and analysis
│   ├── dataloading.py                # HDF5 file parsing utilities
│   └── neuraldata_viz.py             # Visualization utilities
│
├── src/                               # Core implementation
│   ├── models/
│   │   └── baseline.py               # CNN-BiLSTM architecture
│   ├── preprocessing/
│   │   ├── dataloader.py             # PyTorch Dataset and collate functions
│   │   ├── compute_session_stats.py  # Session statistics computation
│   │   ├── session_stats.json        # Precomputed session normalization stats
│   │   └── utils.py                  # Normalization, smoothing, stats utilities
│   └── utils/
│       ├── decoders.py               # Greedy and beam search decoders
│       ├── n_gram.py                 # N-gram language model (CharNGramModel)
│       ├── ngram_3gram.pkl           # Pretrained 3-gram model
│       ├── metrics.py                # WER and evaluation metrics
│       ├── trainingdata_list.py      # Training data paths
│       ├── h5_list_data.json         # HDF5 file paths
│       └── inspect_pklfile.py        # Utility for inspecting pickle files
│
├── scripts/
│   ├── train.py                      # CNN-BiLSTM neural model training
│   ├── train_ngram.py                # N-gram language model training
│   ├── submission.py                 # Generate test predictions and submission CSV
│   ├── models/                       # Trained model checkpoints
│   └── submissions/                  # Output CSV submissions
│
├── diagnose_beam.py                  # Diagnostic tool for beam search decoder testing
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Model Architecture

The solution implements a **CNN-BiLSTM hybrid architecture with integrated n-gram language modeling** for improved decoding accuracy:

### Neural Decoder Architecture

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

### Language Model Integration

This approach combines the neural decoder with a **character-level 3-gram language model**:

- **N-gram Model**: CharNGramModel (trained on all sentences in training set)
- **Decoding Strategy**: Beam search with n-gram scoring
- **Beam Width**: Configurable (typically 10-20 for inference)
- **Language Model Weight**: Adjustable via alpha parameter (0 ≤ alpha ≤ 1)

### Key Features

- **CNN Layers**: Learn spatial-temporal patterns across 512 neural channels
- **BiLSTM**: Capture bidirectional temporal dependencies in neural sequences
- **CTC Loss**: Handle variable-length sequences without frame-level alignment
- **Mixed Precision Training**: Optimized for 6GB+ VRAM using AMP (Automatic Mixed Precision)
- **N-gram Language Model**: Character-level 3-gram model with smoothing
- **Beam Search Decoder**: Combines neural model scores with language model probabilities

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

### Neural Model Configuration

```python
# train.py parameters
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
OPTIMIZER = AdamW (with weight decay: 1e-2)
LOSS = CTC Loss (blank index: 0)
DEVICE = CUDA (with AMP scaling)
```

### Language Model Training

```python
# train_ngram.py parameters
N_GRAM_ORDER = 3  # 3-gram model
TRAINING_DATA = All training set sentences
VOCABULARY = Character-level vocabulary
SMOOTHING = Laplace smoothing (add-one smoothing)
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

**Step 1: Train the neural decoder**
```bash
python scripts/train.py
```
Outputs: Model checkpoints saved to `scripts/models/`

**Step 2: Train the n-gram language model** (optional but recommended)
```bash
python scripts/train_ngram.py
```
Outputs: N-gram model saved to `src/utils/ngram_3gram.pkl`

### Making Predictions

**Generate test predictions with n-gram rescoring**:
```bash
python scripts/submission.py
```

This script performs the following pipeline:
1. Load trained CNN-BiLSTM checkpoint from `scripts/models/best_model.pth`
2. Load n-gram model from `src/utils/ngram_3gram.pkl`
3. Process test sessions from HDF5 files
4. For each test trial:
   - Generate neural model logits
   - Apply beam search decoder with n-gram integration
   - Combine neural scores (CTC) with language model scores
5. Format output as CSV and save to `scripts/submissions/submission.csv`

### Decoder Options

**Greedy Decoder** (fast, no language model):
```python
from src.utils.decoders import greedy_decoder
result = greedy_decoder(logits, tokenizer)
```

**Beam Search Decoder** (slower, with n-gram LM):
```python
from src.utils.decoders import beam_search_decoder
result = beam_search_decoder(logits, tokenizer, ngram_model, 
                            beam_width=10, alpha=0.5)
```
- `beam_width`: Number of hypotheses to track (default: 10)
- `alpha`: Language model weight scaling (0 = pure neural, 1 = full LM weight)

### Expected Output Format

Submission CSV file with chronological test trials (saved to `scripts/submissions/submission.csv`):
```
id,text
0,the quick brown fox
1,jumps over the lazy dog
...
1449,...
```

## Diagnostic Tools

### Beam Search Diagnostic

Test the beam search decoder performance:
```bash
python diagnose_beam.py
```

This utility:
- Creates mock neural logits (500 timesteps × vocabulary size)
- Loads the real n-gram model from `src/utils/ngram_3gram.pkl`
- Runs beam search decoder with beam_width=10
- Reports execution time and output length
- Useful for profiling and debugging decoding pipeline

### N-gram Model Inspection

Inspect contents of pickle files:
```bash
python src/utils/inspect_pklfile.py
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

## Approach Comparison

### Approach #1: CNN + BiGRU (Baseline)
- Single neural decoder with GRU layers
- Greedy decoding only
- Simpler pipeline with fewer components

### Approach #2: CNN + BiLSTM + N-gram (This Approach)
- CNN + BiLSTM neural decoder
- Integrated character-level 3-gram language model
- Beam search decoder with n-gram scoring
- Improved accuracy through language model rescoring
- More computationally intensive but better WER

## N-gram Language Model Details

### CharNGramModel Class

Implementation in `src/utils/n_gram.py`:

**Training**:
- Processes all training sentences (lowercased, stripped)
- Builds n-gram probability tables using context windows
- Uses padding character '~' for sentence boundaries
- Supports Laplace smoothing for unseen n-grams

**Inference**:
- `get_char_log_prob(context, char)`: Returns log-probability of character given context
- Context automatically standardized to (n-1) characters
- Handles OOV characters through smoothing

**Example (3-gram)**:
```
Sentence: "hello"
Padding: "~~hello"
N-grams: ('~~', 'h'), ('~h', 'e'), ('he', 'l'), ('el', 'l'), ('ll', 'o')
Context stored: ~~ → h, ~h → e, he → l, el → l, ll → o
```

## License

The dataset is licensed under **CC0 1.0 Universal** (Public Domain Dedication). Please cite the original work in any publications.

---
