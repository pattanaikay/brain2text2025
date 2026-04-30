# Brain-to-Text 2025: Approach #5 - Transformer + Aero-1-Audio

This repository contains Approach #5 for the **[Kaggle Brain-to-Text 2025 Competition](https://www.kaggle.com/competitions/brain-to-text-25)**, which aims to decode speech directly from intracortical neural activity. This approach leverages modern multimodal LLMs by using a Transformer encoder for neural data preprocessing and integrating with the Aero-1-Audio language model.

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
approach #5- Transformer + Aero-1-Audio/
├── scripts/
│   ├── train.py                      # Training pipeline with SSL pretraining and supervised fine-tuning
│   ├── submission.py                 # Generate test predictions and submission CSV
│   ├── models/                       # Trained model checkpoints (LoRA adapters)
│   └── submissions/                  # Output CSV submissions
│
├── src/                               # Core implementation
│   ├── models/
│   │   └── baseline.py               # Transformer encoder + Aero-1-Audio multimodal model
│   ├── preprocessing/
│   │   └── dataloader.py             # PyTorch Dataset and collate functions
│   └── utils/
│       └── metrics.py                # WER and CER evaluation metrics
│
├── requirements.txt                  # Python dependencies (transformers, peft, bitsandbytes)
└── README.md                         # This file
```

## Model Architecture

This approach implements a **Transformer-based neural encoder integrated with Aero-1-Audio multimodal LLM** for end-to-end brain-to-text decoding:

### Neural Encoder Architecture

```
Input: (Batch, Time, 512 neural channels)
    ↓
Linear Projection
  - Projects 512 channels to 384-dim embeddings
    ↓
Transformer Encoder
  - 7 layers
  - 6 attention heads
  - 1024-dim feedforward
  - Dropout: 0.1
    ↓
Layer Normalization
    ↓
Output: (Batch, Time, 384-dim embeddings)
```

### Multimodal LLM Integration

```
Neural Embeddings (384-dim)
    ↓
MLP Projector
  - Projects to LLM embedding space (1536-dim for Aero-1-Audio)
  - Hidden layers with ReLU activation
    ↓
Prepend Prompt ("decode the above neural activity into an English sentence:")
    ↓
Aero-1-Audio LLM (1.5B parameters)
  - LoRA fine-tuning (rank=16, alpha=32)
  - 4-bit quantization for memory efficiency
  - Causal language modeling
    ↓
Output: Generated text
```

### Key Components

- **NeuralEncoder**: Transformer encoder with 7 layers, 6 heads, 384-dim embeddings
- **MLPProjector**: Maps 384-dim encoder output to 1536-dim LLM input space
- **Aero-1-Audio LLM**: Multimodal language model (audio-tuned) with LoRA fine-tuning
- **4-bit Quantization**: Uses BitsAndBytes for memory-efficient inference
- **LoRA Adaptation**: Low-rank fine-tuning of Q, K, V, O projection matrices

### Key Features

- **Transformer Architecture**: Self-attention mechanisms for capturing long-range dependencies in neural sequences
- **Multimodal LLM**: Leverages a pretrained language model optimized for audio/speech understanding
- **Parameter-Efficient Fine-Tuning**: LoRA enables rapid adaptation with minimal memory overhead
- **4-bit Quantization**: Reduces model size and inference time while maintaining quality
- **Self-Supervised Pretraining**: Optional SSL phase for learning robust neural representations
- **Prompt Engineering**: Uses task-specific prompts to guide LLM output

## Training Pipeline

### Configuration

```python
# train.py parameters
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
OPTIMIZER = AdamW (with weight decay: 0.01)
SCHEDULER = ReduceLROnPlateau (factor=0.5, patience=10)
DEVICE = CUDA (with 4-bit quantization)

# Neural Encoder Config
ENCODER_DIM = 384
ENCODER_HEADS = 6
ENCODER_LAYERS = 7
ENCODER_DROPOUT = 0.1

# LoRA Config
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Two-Stage Training Process

#### Stage 1: Self-Supervised Pretraining (Optional)

```
Purpose: Learn robust neural representations from unlabeled data
Method: Masked neural activity prediction
  - Randomly mask 15% of neural time steps
  - Train encoder to predict masked regions
  - Loss: MSE between masked and actual neural embeddings
Duration: ~50 epochs
Output: Pretrained neural encoder weights
```

#### Stage 2: Supervised Fine-Tuning

```
Purpose: Fine-tune model end-to-end with supervision
Method: Causal language modeling with LLM
  - Forward pass through neural encoder
  - Project encoder output to LLM space
  - Concatenate with prompt and labels
  - Compute language modeling loss
  - Backprop through LoRA adapters
Duration: ~50 epochs
Output: LoRA adapter weights (saved to scripts/models/)
```

### Data Loading

- **Custom PyTorch Dataset**: Handles variable-length neural sequences and text
- **Collate Function**: Dynamically pads sequences to max length in batch
- **Pin Memory**: Enabled for faster GPU transfer
- **Workers**: 4 parallel data loading processes

## Usage

### Installation

```bash
# Clone repository
git clone <repo_url>
cd Brain2Text2025/approach\ #5-\ Transformer\ +\ Aero-1-Audio

# Install dependencies
pip install torch>=2.0.0 transformers>=4.36.0
pip install peft>=0.7.0 bitsandbytes>=0.42.0
pip install h5py scipy numpy tqdm jiwer pandas
pip install accelerate>=0.25.0  # For multi-GPU training
```

### Running Training

**Step 1: Optional SSL Pretraining** (improves convergence)
```bash
python scripts/train.py --ssl_pretrain --epochs 50
```
Outputs: Pretrained encoder weights

**Step 2: Supervised Fine-Tuning**
```bash
python scripts/train.py --epochs 50
```
Outputs: LoRA adapter weights saved to `scripts/models/lora_adapter/`

### Making Predictions

**Generate test predictions**:
```bash
python scripts/submission.py
```

This script performs the following pipeline:
1. Load pretrained neural encoder from checkpoint
2. Load Aero-1-Audio model with 4-bit quantization
3. Load LoRA adapters from `scripts/models/lora_adapter/`
4. Process test sessions from HDF5 files
5. For each test trial:
   - Generate neural encoder embeddings
   - Project to LLM space
   - Prepend task prompt
   - Generate output text using LLM
6. Format output as CSV and save to `scripts/submissions/submission.csv`

### Expected Output Format

Submission CSV file with chronological test trials:
```
id,text
0,the quick brown fox
1,jumps over the lazy dog
...
1449,...
```

## Preprocessing Pipeline

### 1. Z-Score Normalization (Per-Session)

Neural signals are inherently non-stationary. Due to electrode impedance drift and physical array shifts, channel baselines change between recording sessions:

$$Z = \frac{x - \mu}{\sigma}$$

- Computed across all trials in a session for each channel
- Handles dead channels by adding epsilon (1e-8) to prevent division by zero

### 2. Gaussian Smoothing

Neural firing is stochastic. Gaussian filtering the time series acts as a low-pass filter to reveal underlying motor intent:

- **Sigma**: 1.5 (covers ~30-40ms at 20ms bins)
- Applied per-channel along the time axis
- Reduces jitter while preserving critical neural dynamics

## Model Capabilities

### Advantages of Approach #5

1. **Transformer Self-Attention**: Better captures long-range dependencies compared to RNN-based approaches
2. **Multimodal LLM**: Aero-1-Audio is trained on audio/speech, making it well-suited for speech motor cortex data
3. **Parameter Efficiency**: LoRA allows fine-tuning without storing full model weights
4. **Scalability**: 4-bit quantization enables inference on limited hardware
5. **End-to-End Learning**: Direct mapping from neural activity to text with no intermediate phoneme representations

### Limitations

1. **Computational Cost**: Requires significant GPU memory for LLM loading/inference
2. **Inference Speed**: Slower than RNN-based approaches due to transformer complexity
3. **Data Requirements**: May need more training data to fully utilize LLM capacity
4. **Hyperparameter Sensitivity**: More tuning parameters (encoder layers, heads, LoRA rank, etc.)

## Approach Comparison

### Approach #1: CNN + BiGRU (Baseline)
- Simple, fast RNN-based architecture
- Greedy decoding
- Lowest memory requirements

### Approach #2: CNN + BiLSTM + N-gram (Language Modeling)
- LSTM-based with explicit n-gram LM
- Beam search decoding
- Better accuracy through linguistic priors

### Approach #5: Transformer + Aero-1-Audio (This Approach)
- Modern transformer encoder + multimodal LLM
- Leverages pretrained speech understanding
- Best accuracy potential but higher computational cost
- Parameter-efficient fine-tuning with LoRA

## References

### Primary Citation

Card, N., Wairagkar, M., Iacobacci, C., et al. (2025). "Brain-to-text '25." Kaggle Competition. https://kaggle.com/competitions/brain-to-text-25

### Related Work

- **Aero-1-Audio**: LMMS-Lab multimodal LLM optimized for audio (https://github.com/LMM-Lab/Aero)
- **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- **Original NEJM Publication**: Card et al. "An Accurate and Rapidly Calibrating Speech Neuroprosthesis" (2024) *New England Journal of Medicine*
- **Previous Challenge**: [Brain-to-Text 2024](https://eval.ai/web/challenges/challenge-page/2099/evaluation)
- **Research Lab**: [UC Davis Neuroprosthetics Lab](https://neuroprosthetics.science/) (part of [BrainGate Consortium](https://www.braingate.org/))

## Resources

- **Baseline Algorithm**: https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text
- **Dataset (Dryad)**: https://doi.org/10.5061/dryad.dncjsxm85
- **Competition Discussion**: Kaggle discussion board
- **Contact**: nscard@health.ucdavis.edu

## License

The dataset is licensed under **CC0 1.0 Universal** (Public Domain Dedication). Please cite the original work in any publications.

---
