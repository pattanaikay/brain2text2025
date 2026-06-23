# ECoG Autoresearch: Laptop-Scale Neural Benchmarking

**[← Back to Main README](../README.md)**

🔬 **PARALLEL AUTORESEARCH PLATFORM**: Small, reproducible benchmark for ECoG finger-flexion regression with compact CNN, Transformer, NeuroMoE, HRM, and HRM+MoE models. Designed for agent-driven autoresearch with a fixed benchmark harness.

---

## 🚀 Quick Start

### 1. Setup Environment
On Windows, the most reliable path is WSL2 with CUDA, but native Windows can work if PyTorch CUDA is installed correctly.

```powershell
cd ecog-autoresearch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA, install the PyTorch build recommended for your GPU from https://pytorch.org/get-started/locally/.

### 2. Prepare Data
```powershell
python prepare_data.py --subject 1 --sfreq 200 --window-sec 1.5 --stride-sec 0.25
```

This creates `data/ecog_fingerflex_subject1.npz` with train/validation/test windows.

### 3. Run Benchmarks
```powershell
python benchmark.py --model cnn --budget-minutes 3 --batch-size 0
python benchmark.py --model transformer --budget-minutes 3
python benchmark.py --model neuromoe --budget-minutes 3
python benchmark.py --model hrm --budget-minutes 3
```

Each run appends to `results.tsv` and writes run artifacts under `runs/<run_id>/`.

### 4. Plot Results
```powershell
python plot_results.py
```

Plots are written to `plots/`.

---

## 🔄 Key Changes & Updates

- ✅ **Compact Benchmark Dataset**: BCI Competition IV Dataset 4 (64-channel ECoG, 5 finger-flexion targets)
- ✅ **Multiple Model Variants**: CNN, Transformer, NeuroMoE, HRM, HRM+MoE
- ✅ **Laptop-Friendly**: Optimized for ~6 GB VRAM / 24 GB RAM
- ✅ **Autoresearch Ready**: Fixed benchmark harness; agents modify only `train.py`
- ✅ **Results Tracking**: TSV-based results with automatic plotting
- ✅ **Hardware Agnostic**: Works on CUDA, CPU, and Apple Silicon
- ✅ **Run Artifacts**: Complete traceability with model checkpoints and metrics
- ✅ **Pearson Correlation Metrics**: Primary metric for validation performance

---

## Overview

Small, reproducible benchmark for ECoG finger-flexion regression with compact CNN, Transformer, NeuroMoE, HRM, and HRM+MoE models.

The default target is BCI Competition IV Dataset 4 / Miller finger-flexion ECoG. The harness is designed for a laptop with about 6 GB VRAM and 24 GB RAM.

## Setup

Install Python 3.10+ first. On Windows, the most reliable path is often WSL2 with CUDA, but native Windows can work if PyTorch CUDA is installed correctly.

```powershell
cd ecog-autoresearch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For CUDA, install the PyTorch build recommended for your GPU from https://pytorch.org/get-started/locally/.

## Prepare Data

```powershell
python prepare_data.py --subject 1 --sfreq 200 --window-sec 1.5 --stride-sec 0.25
```

This creates `data/ecog_fingerflex_subject1.npz` with train/validation/test windows.

## Run Benchmarks

```powershell
python benchmark.py --model cnn --budget-minutes 3 --batch-size 0
python benchmark.py --model transformer --budget-minutes 3
python benchmark.py --model neuromoe --budget-minutes 3
python benchmark.py --model hrm --budget-minutes 3
```

Each run appends to `results.tsv` and writes run artifacts under `runs/<run_id>/`.

Use `--batch-size 0` for the harness default: 64 on CUDA and 32 on CPU.

## Plot Results

```powershell
python plot_results.py
```

Plots are written to `plots/`.

## Autoresearch Rule

For agent-driven loops, keep `prepare_data.py`, `benchmark.py`, and `plot_results.py` fixed. Let the agent modify only `train.py`, run `benchmark.py`, and keep a change only if mean validation Pearson correlation improves without breaking the benchmark contract.
