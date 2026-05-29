# Brain-to-Text 2025: Multi-Track Neural Decoding Research

A comprehensive research repository exploring multiple architectures and techniques for decoding intended speech directly from intracortical neural activity. This project encompasses seven major work streams: five primary neural decoding approaches, a parallel autoresearch benchmark, and a production training framework.

**Competition**: [Kaggle Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25)  
**Dataset**: 10,948 sentences from participant T15 across 45 recording sessions (256 microelectrodes, 512 neural features)  
**Baseline WER**: 36.73% (BIT + Qwen2.5-1.5B)  
**Target WER**: 10%

---

## Quick Reference: Project Structure

This repository is organized into **independent work streams**, each with its own README for detailed documentation. Start here for an overview, then navigate to the folder of interest.

| Folder | Purpose | Status | Key Metric |
|--------|---------|--------|------------|
| **[Approach #1: CNN + BiGRU](./approach%20%231-%20CNN%20+%20BiGRU/)** | Baseline CNN-BiGRU with CTC loss | ✅ Baseline | WER: Baseline |
| **[Approach #2: CNN + BiLSTM + n-gram](./approach%20%232-%20CNN%20+%20BiLSTM%20+%20ngram/)** | CNN-BiLSTM with n-gram language model | ✅ Improved | WER: Better |
| **[Approach #5: Transformer + Aero-1-Audio](./approach%20%235-%20Transformer%20+%20Aero-1-Audio/)** | Transformer encoder + multimodal LLM (BIT v1) | ✅ Production | WER: Best |
| **[Approach #6: Transformer + Aero-1-Audio + Diphones](./approach%20%236-%20Transformer%20+%20Aero-1-Audio%20+%20Diphones/)** | BIT with diphone linguistic support (BIT v2) | 🔧 Experimental | WER: Excellent |
| **[Approach #7: NeuroMoE](./approach%20%237-%20NeuroMoE/)** | Mixture-of-Experts + Regional Experts framework | 🚀 SOTA | WER: SOTA Target |
| **[Brain2Text Model Training](./brain2text-modeltraining/)** | **MAIN PRODUCTION**: BIT framework reference implementation | ✅ Active | WER: Monitored |
| **[Brain2Text Experiments](./brain2text-experiments/)** | **MAIN RESEARCH**: 25-experiment framework with tracks A–E | ✅ Active | 25+ Experiments |
| **[Brain2Text Multi-Arch](./brain2text-modeltraining-multiarchitectures/)** | Multi-architecture exploration & analysis | ✅ Active | Comparative Analysis |
| **[ECoG Autoresearch](./ecog-autoresearch/)** | **PARALLEL**: Laptop-scale agent-driven benchmarking (finger flexion) | ✅ Active | Pearson r |
| **[DietCorp + ZenBrain](./dietcorp-zenbrain-tta-research/)** | **RESEARCH**: Edge-device feasibility (Coral NPU simulation) | 🔬 Investigation | On-Device WER |

---

## Work Stream Descriptions

### Primary Competition Approaches (Approach #1–7)

These folders explore different neural architectures for brain-to-text decoding on the Kaggle competition dataset.

#### **Approach #1: CNN + BiGRU** [(details)](./approach%20%231-%20CNN%20+%20BiGRU/README.md)
- Baseline convolutional encoder with bidirectional GRU
- Uses CTC loss and per-session Z-score normalization
- Fast training, establishes performance floor
- **Status**: ✅ Complete and validated

#### **Approach #2: CNN + BiLSTM + n-gram** [(details)](./approach%20%232-%20CNN%20+%20BiLSTM%20+%20ngram/README.md)
- Improves on #1 by adding LSTM and language model rescoring
- Beam search decoding with 3-gram language model
- Incorporates linguistic priors to improve WER
- **Status**: ✅ Complete; improved baseline established

#### **Approach #5: Transformer + Aero-1-Audio (BIT v1)** [(details)](./approach%20%235-%20Transformer%20+%20Aero-1-Audio/README.md)
- Time-patched Transformer encoder (RoPE, 7 layers, 384 dim)
- Multimodal LLM (Aero-1-Audio-1.5B) decoder
- 3-phase training: SSL pretraining → end-to-end fine-tuning → optional CTC
- Contrastive loss for modality alignment
- Session-specific read-in layers for drift correction
- **Status**: ✅ Production-ready; strong performance

#### **Approach #6: Transformer + Aero-1-Audio + Diphones (BIT v2)** [(details)](./approach%20%236-%20Transformer%20+%20Aero-1-Audio%20+%20Diphones/README.md)
- Builds on #5 with diphone-level linguistic constraints
- Enhanced phonetic modeling for better accuracy
- Experimental variant exploring linguistic granularity
- **Status**: 🔧 In development; shows promise

#### **Approach #7: NeuroMoE (SOTA)** [(details)](./approach%20%237-%20NeuroMoE/README.md)
- Advanced Mixture-of-Experts architecture (SSMoE + Regional)
- 6 specific + 2 shared experts with Top-K routing (K=2)
- 8 brain-region processing modules (64 channels each)
- Integrates EEGMoE (Gao et al., 2026) + BrainStack (Zhao et al., 2026)
- Multimodal loss with auxiliary load-balancing
- **Status**: 🚀 Most advanced; targets SOTA performance

---

### Main Production & Research Frameworks

#### **Brain2Text Model Training** [(details)](./brain2text-modeltraining/README.md)
The **primary reference implementation** of the BIT framework. Production-ready code for end-to-end training pipelines.

- **Phase 1**: Self-supervised pretraining (Masked Neural Modeling)
- **Phase 2**: End-to-end fine-tuning with CE + Contrastive loss
- **Phase 3**: Optional CTC-based phoneme recognition
- **Checkpoint management**, metric tracking, evaluation utilities
- **Documentation**: Comprehensive technical specification and debugging guides
- **Status**: ✅ Phase 1 complete (PER=0.5202); Phase 2 in debug

**Key Features**:
- Full training pipeline with validation monitoring
- Multiple loss configurations
- Results logging to SQLite leaderboard
- Session statistics computation for normalization

#### **Brain2Text Experiments** [(details)](./brain2text-experiments/README.md)
The **research-grade experimentation framework** with 25+ experiments across 5 tracks (A–E).

- **Registry system**: Central `registry.yaml` for all experiment definitions
- **Profile system**: Toy (local, ~20 min) vs. Full (cloud A100, 150 epochs) profiles
- **Three-step progression**: Shape tests → toy run → full run (enforced)
- **Composed losses**: Multi-loss ablation via forward hooks
- **Modular stages**: Swappable encoders, projectors, decoders, losses
- **Leaderboard tracking**: SQLite database with WER, metrics, and reproducibility hashes

**Experiment Tracks**:
- **Track A**: Pretraining modality analysis (CKA, perplexity, probing)
- **Track B**: Encoder variants (BIT, Conformer, HRM, Mamba, MoE, ZenBrain)
- **Track C**: Decoder LLM variants (Qwen, Phi, Whisper-Qwen)
- **Track D**: Loss function ablations (CTC, contrastive, topological)
- **Track E**: Projector design variants (MLP, deep MLP, gated, QFormer)

#### **Brain2Text Multi-Architectures** [(details)](./brain2text-modeltraining-multiarchitectures/README.md)
Exploration and comparative analysis of multiple architecture variants.

- Multi-architecture benchmarking
- Comparative performance metrics
- Reference implementations of emerging techniques
- Analysis tools and visualization utilities
- **Status**: ✅ Active research platform

---

### Parallel Work Streams

#### **ECoG Autoresearch Benchmark** [(details)](./ecog-autoresearch/README.md)
A **laptop-scale, agent-driven autoresearch harness** for rapid prototyping on a compact benchmark.

- **Dataset**: BCI Competition IV Dataset 4 (64-channel ECoG, 5 finger-flexion targets)
- **Primary Metric**: Validation mean Pearson correlation
- **Models**: CNN, Transformer, NeuroMoE, HRM, HRM+MoE variants
- **Hardware**: Optimized for ~6 GB VRAM / 24 GB RAM (laptop-friendly)
- **Autoresearch**: Agents modify only `train.py`; benchmark harness (`prepare_data.py`, `benchmark.py`, `plot_results.py`) stays fixed
- **Change Policy**: Keep changes only if validation Pearson correlation improves
- **Status**: ✅ Active autoresearch platform

**Purpose**: Validates neural architecture designs on a compact, reproducible benchmark before scaling to full speech decoding tasks.

#### **DietCorp + ZenBrain Feasibility Study** [(details)](./dietcorp-zenbrain-tta-research/README.md)
Investigation of edge-device feasibility for real-time BCI deployment.

- **Base Architecture**: DietCorp-Compact (9.4M params, 364 MFLOPS, causal Transformer)
- **Extension**: ZenBrain-inspired multi-tier memory layer
- **Target Hardware**: Google Coral NPU (simulated and real)
- **Simulation Instrument**: Cycle-accurate Verilator simulator for architectural analysis
- **Goals**: 
  1. Characterize feasibility under realistic edge constraints (int8, static shapes, limited memory)
  2. Project hardware requirements for production deployment
  3. Surface design constraints early
  4. Produce communicable artifacts for thesis documentation
- **Status**: 🔬 Feasibility audit complete; entering reproduction phase

**Key Context**: Coral is used as a simulation microscope for edge NPU behavior, not as the deployment target. Eventual deployment targets are phones and laptops (Apple Neural Engine, Qualcomm Hexagon, Google Tensor TPU).

---

## Getting Started

### Option 1: Explore a Specific Approach
Pick an approach folder and read its README:
```bash
cd "approach #7- NeuroMoE"
cat README.md
pip install -r requirements.txt
```

### Option 2: Use the Main Production Framework
Start with [Brain2Text Model Training](./brain2text-modeltraining/README.md):
```bash
cd brain2text-modeltraining
pip install -r requirements.txt
python scripts/train_ssl.py --epochs 50
```

### Option 3: Run the Research Experiments Framework
See [Brain2Text Experiments](./brain2text-experiments/README.md):
```bash
cd brain2text-experiments
pip install -r requirements.txt
python -m pytest tests/test_stage_shapes.py -v
python run.py --expt B0_baseline --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
```

### Option 4: Run the Autoresearch Benchmark
See [ECoG Autoresearch](./ecog-autoresearch/README.md):
```bash
cd ecog-autoresearch
pip install -r requirements.txt
python prepare_data.py --subject 1
python benchmark.py --model auto --budget-minutes 5
python plot_results.py
```

### Option 5: Investigate Edge Deployment
See [DietCorp + ZenBrain](./dietcorp-zenbrain-tta-research/README.md):
```bash
# See folder README for simulation and analysis setup
```

---

## Key Documentation Files

| Document | Scope | Location |
|----------|-------|----------|
| **Approach-specific READMEs** | Detailed setup, architecture, results for each approach | Each approach folder |
| **Brain2Text Model Training README** | Production BIT framework and training pipeline | `brain2text-modeltraining/` |
| **Brain2Text Experiments README** | 25-experiment research framework and registry system | `brain2text-experiments/` |
| **ECoG Autoresearch README** | Agent-driven benchmarking for finger-flexion | `ecog-autoresearch/` |
| **DietCorp README** | Edge-device feasibility studies and Coral NPU simulation | `dietcorp-zenbrain-tta-research/` |
| **BCI Incremental Roadmap** | Historical development from RNN baseline to BIT framework | `bci-incremental-roadmap.md` |
| **Technical Specifications** | Deep dives on architecture, loss functions, training pipelines | `*/COMPREHENSIVE_TECHNICAL_SPEC.md` |

---

## Architecture Comparison at a Glance

| Feature | #1 | #2 | #5 | #6 | #7 |
|---------|----|----|----|----|-----|
| **Encoder** | CNN | CNN | Transformer | Transformer | Transformer |
| **Decoder** | Greedy | Beam+n-gram | LLM | LLM+Diphone | LLM |
| **Memory/Routing** | — | — | — | — | SSMoE + Regional |
| **Pretraining** | None | None | Optional SSL | Optional SSL | Optional SSL |
| **Drift Handling** | Norm | Norm | Read-in | Read-in | Read-in |
| **Training Time** | 2-4 hrs | 3-5 hrs | 6-10 hrs | 6-10 hrs | 8-12 hrs |
| **Expected WER** | Baseline | Better | Best | Excellent | **SOTA** |
| **Complexity** | Low | Medium | High | High+ | Very High |
| **Status** | ✅ Complete | ✅ Complete | ✅ Prod | 🔧 Exp | 🚀 SOTA |

---

## Known Issues & Status

### Brain2Text Model Training
- **Phase 1 (SSL)**: ✅ Complete (PER=0.5202)
- **Phase 2 (E2E)**: 🔧 In debug (CE loss anomaly, padding masking issues)
- **Phase 3 (CTC)**: ✅ Complete and ready for testing
- **Details**: See [debug_e2e_training.md](./brain2text-modeltraining/debug_e2e_training.md)

### Brain2Text Experiments
- **Registry**: ✅ Complete (25+ experiments defined)
- **Stage Builders**: ✅ Complete (all encoders, projectors, decoders)
- **Shape Testing**: ✅ Complete
- **Loss Composition**: ✅ Complete
- **Full Run Tracking**: ✅ Complete (leaderboard.sqlite)

---

## Contributing & Collaboration

When adding to this repository:

1. **New Approach**: Create a folder with its own `README.md`, `requirements.txt`, and `scripts/` + `src/` structure
2. **Experiment Addition**: Add entry to `brain2text-experiments/registry.yaml` and corresponding spec YAML
3. **Code Changes**: Keep changes scoped to the work stream; document in the folder's README
4. **Documentation**: Update the relevant folder's README and this main README's quick reference table

---

## References

- **Competition**: [Kaggle Brain-to-Text 2025](https://www.kaggle.com/competitions/brain-to-text-25)
- **Dataset**: [Dryad: Brain-to-Text Dataset](https://doi.org/10.5061/dryad.dncjsxm85)
- **Papers**:
  - Zhang et al. (2025). "A cross-species neural foundation model for end-to-end speech decoding" (BIT Framework)
  - Feghhi et al. (2025). "DietCorp-Compact: Efficient test-time adaptation for neural decoding" (arXiv:2507.02800)
  - Gao et al. (2026). "EEGMoE: Mixture-of-Experts for EEG Analysis"
  - Zhao et al. (2026). "BrainStack: Regional Brain Processing Architecture"
  - Bering et al. (2026). "ZenBrain: Neuroscience-grounded memory architecture for autonomous AI" (arXiv:2604.23878)
- **Tools**:
  - [Aero-1-Audio](https://github.com/LMM-Lab/Aero): Multimodal LLM
  - [LoRA](https://arxiv.org/abs/2106.09685): Low-rank adaptation (Hu et al., 2021)
  - [Google Coral](https://coral.ai/): Edge TPU & NPU platforms
- **Lab**: [UC Davis Neuroprosthetics Lab](https://neuroprosthetics.science/)

---

**Last Updated**: May 29, 2026  
**Repository Status**: Active multi-track research  
**Primary Contact**: Brain-to-Text 2025 Research Team
