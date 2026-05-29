# DietCorp + ZenBrain Edge-Device Feasibility Study

Research investigation into the feasibility of real-time, on-device neural speech decoding for edge deployment. This project uses **Google Coral NPU simulators** as an instrumentation platform to characterize how DietCorp-Compact (a lightweight causal Transformer-based neural decoder) behaves under realistic edge constraints, and explores potential extensions using ZenBrain-inspired memory layers.

**Project Status**: 🔬 Feasibility audit complete; entering reproduction phase  
**Date Started**: May 28, 2026  
**Target Hardware**: Google Coral NPU (first-gen Edge TPU, Coral NPU reference IP)  
**Deployment Goal**: Phone/laptop NPU execution for ALS/brainstem-stroke BCIs

---

## Project Overview

### The Clinical Context
The goal is a patient-facing speech brain-computer interface (BCI) for people with ALS (Amyotrophic Lateral Sclerosis) or brainstem stroke. Both conditions destroy motor pathways while preserving cognition, leaving patients unable to speak despite knowing what they want to say. A BCI bypasses the motor pathway and decodes intended speech directly from intracortical neural recordings into text or synthesized voice.

A clinically useful system needs **four properties simultaneously**:
1. **Low word error rate (WER)** — text the patient can actually read
2. **Real-time streaming decoding** — output appears as the patient attempts to speak
3. **On-device feasibility** — runs on consumer hardware (phone, tablet, laptop, small edge module) without cloud connectivity
4. **Robustness to distribution shift** — adapts to day-to-day neural drift without manual re-calibration

### The Technical Approach

**Base Architecture: DietCorp-Compact**  
(Feghhi et al., 2025, arXiv:2507.02800)
- Unidirectional causal Transformer (5 layers, hidden dim 384, 9.4M parameters)
- Decodes 256-channel intracortical microelectrode array activity into phonemes at 10 Hz
- Aggressive time-masking during training (~53% of each trial masked) delays overfitting
- **DietCORP test-time adaptation (TTA)**: Per trial, generate 64 time-masked augmentations, decode pseudo-label via beam search + n-gram LM, take one AdamW step on CTC loss — updating *only the patch-embedding module*
- **Reported performance**: 12.17% WER with 3-gram LM (20% relative improvement over baseline), 18 ms TTA latency per trial
- **Streaming budget**: Each 100 ms patch's forward pass must complete in well under 100 ms; TTA happens between patches

**Proposed Extension: ZenBrain-Inspired Memory Layer**  
(Bering, 2026, arXiv:2604.23878)
- Multi-tier memory architecture with different time constants and consolidation rules
- Hypothesis: Layered memory + adaptation should improve robustness under distribution shift beyond single-shot TTA
- Biological plausibility enables future BCI research to build on the framework
- *Implementation note*: ~80% of ZenBrain lives on host CPU; only working-memory KV cache and embedding/recall operations are NPU candidates

### Why Coral?

**Coral is an instrument, not a deployment target.**

- Simulates realistic edge-NPU constraints: int8 quantization, static shapes, limited control flow, bounded on-chip memory, no on-device backprop
- Represents the strictness envelope of production deployments (Apple Neural Engine, Qualcomm Hexagon, Google Tensor TPU)
- Only edge NPU with public, cycle-accurate simulators available today
- Two complementary platforms:
  - **First-gen Edge TPU** (USB Accelerator, Dev Board Mini): Closed proprietary ASIC; gives real wall-clock int8 latency today
  - **Coral NPU** (2025–2026): Open RISC-V reference IP; gives cycle-level architectural visibility via Verilator simulator

---

## Project Structure

```
dietcorp-zenbrain-tta-research/
├── Coral_Feasibility_DietCorp_ZenBrain.md      # Full feasibility audit & op compatibility analysis
├── Progress_Summary_Scope_and_Plan.md          # Investigation modes & experimentation roadmap
├── README.md                                    # This file
│
├── simulations/                                 # Coral simulator integration (future)
│   ├── tflite_converter/                        # PyTorch → TFLite conversion pipeline
│   ├── coral_edge_tpu/                          # First-gen Edge TPU experiments
│   └── coral_npu/                               # Coral NPU + Verilator experiments
│
├── models/                                      # DietCorp implementation & weights
│   ├── dietcorp_compact.py                      # Core model implementation
│   ├── attention.py                             # Causal attention + RoPE + T5 relative bias
│   ├── patch_embedding.py                       # Patch embedder (drift-adapted)
│   ├── tta_adaptation.py                        # DietCORP TTA procedure
│   └── checkpoints/                             # Pre-trained weights
│
├── memory/                                      # ZenBrain memory layer experiments (future)
│   ├── kv_cache.py                              # Fixed-shape working memory KV cache
│   ├── consolidation.py                         # Memory consolidation rules
│   └── neuromodulator.py                        # Neuromodulator dynamics (simplified)
│
├── data/                                        # Test datasets & reference benchmarks
│   ├── neural_test_samples.h5                   # Small validation set (10 trials)
│   └── baseline_wer.txt                         # Reference WER from GPU training
│
├── analysis/                                    # Results analysis & visualization
│   ├── plot_latency_breakdown.py                # Per-layer timing analysis
│   ├── compare_int8_accuracy.py                 # Accuracy degradation from quantization
│   └── results/                                 # Generated plots & metrics
│
└── docs/                                        # Documentation & design notes
    ├── arch_diagram.md                          # Architecture overview
    ├── simulator_notes.md                       # Verilator simulator guide
    └── hardware_projection.md                   # Derived hardware requirements spec
```

---

## Key Findings from Feasibility Audit

### Finding 1: DietCorp-Compact is Simulatable with Clear Constraints

**The Good:**
- Model is small (9.4M params, 364 MFLOPS, ~10 MB int8), causal, and streaming
- Dominated by matmul/softmax/LayerNorm operations (clean fit for NPU hardware)
- Fits comfortably on both Edge TPU (5 MB typical) and Coral NPU (designed for ambient sensing)

**The Hard Part:**
- DietCORP's per-trial backward pass + Adam step requires **on-device training**
- Neither Edge TPU nor Coral NPU runtime currently supports on-device backprop
- **Clean separation**: Forward pass (inference) on NPU, gradient computation + parameter update on host CPU
- This mirrors production deployment architecture on phone NPUs (TTA gradient steps also run on host CPU)

**Simulator Accuracy:**
- Verilator (cycle-accurate): Gives genuine per-cycle timing and AXI memory transaction counts
- MPACT-CoralNPU (functional): Fast but less detailed
- **Caveat**: Matrix execution unit "under development" as of Jan 2026; latency for ops not yet optimized may be 2–5× off

### Finding 2: Bolting ZenBrain On Top is ~80% Host-Side

**What Stays on Host:**
- Knowledge-graph storage
- BM25 + embedding retrieval (mostly database operations)
- FSRS (Forgetting Space Retrieval) scheduling
- Neuromodulator dynamics & reconsolidation snapshots
- TripleCopy stores with multiple τ decay rates

**What Goes on NPU:**
- Working-memory KV cache (fixed-shape, co-located with DietCorp patch-embedding)
- Embedding/similarity computations for `recall(query)` (only if fused with encoder)

**Implication**: Simulation work can explore architectural variants beyond static memory — try dynamic graph-attention, try 2-tier static memory — because most constraints are software, not hardware.

### Finding 3: Useful Metrics are Narrower Than They Look

**What the Verilator Sim Gives (Reliable):**
- Cycle counts per layer
- Per-op timing breakdowns
- AXI memory transactions & on-chip vs off-chip traffic
- Memory-bandwidth pressure

**What It Does NOT Give (Must Measure Separately):**
- Wall-clock latency on real silicon
- Power consumption, dynamic energy, thermal effects
- DRAM contention with other SoC workloads
- Accuracy degradation from int8 quantization (captured offline at conversion time)
- TTA convergence on-device (is a full software experiment, not a simulator question)

---

## Investigation Roadmap

### Phase 1: Baseline Characterization (May–June 2026)
**Goal**: Measure DietCorp-Compact inference on Coral; compare to GPU baseline.

**Milestones:**
1. Convert PyTorch DietCorp → TFLite float graph
2. Post-training int8 quantization with neural data representative set
3. Compile to Edge TPU (first-gen) + Coral NPU
4. Measure latency, accuracy, memory on real hardware (USB Accelerator) + Verilator simulator
5. Produce latency breakdown per layer & per operation

**Expected Outputs:**
- Latency table: CPU vs Edge TPU vs Coral NPU
- Accuracy table: FP32 vs int8 (quantization loss)
- Memory footprint & peak SRAM usage
- Hardware requirements projection (based on real measurements)

### Phase 2: ZenBrain Exploration (June–July 2026)
**Goal**: Prototype memory augmentation and measure its impact on streaming latency.

**Milestones:**
1. Implement static working-memory KV cache (fixed shape, 100 ms window)
2. Integrate with DietCorp patch-embedding via forward fusion
3. Simulate on Coral NPU Verilator
4. Measure latency & accuracy impact of memory layer
5. Design trade-offs: memory size vs. latency vs. adaptation performance

**Expected Outputs:**
- Latency vs. memory-size curves
- Accuracy improvement from memory-augmented TTA
- Feasibility report: "Can ZenBrain-lite fit in streaming budget?"

### Phase 3: Hardware Requirements Projection (July–August 2026)
**Goal**: Synthesize measurements into a spec sheet for production deployment hardware evaluation.

**Deliverables:**
- MAC/sec requirements
- On-chip SRAM requirements
- Off-chip bandwidth budget
- Op-set requirements (softmax, LayerNorm, causal mask, relative position bias)
- Latency targets (must complete <100 ms per patch)
- Power envelope estimate

**Use**: Any candidate production substrate (Apple Neural Engine, Qualcomm Hexagon, Google Tensor, Intel AI Boost) can be evaluated against this spec to determine feasibility.

### Phase 4: Thesis Documentation (August–September 2026)
**Goal**: Produce communicable artifacts for thesis chapter & defense.

**Outputs:**
- Latency breakdown visualizations
- Accuracy-latency trade-off plots
- Hardware requirements spec sheet
- Recommendations for production deployment
- Limitations & future work

---

## Quick Start: Run a Simulation

(Once Phase 1 is complete)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
# Includes: torch, tensorflow, onnx, onnx-tf, tflite-runtime, numpy, matplotlib, pandas
```

### 2. Convert and Quantize DietCorp
```bash
python simulations/tflite_converter/convert_to_tflite.py \
    --checkpoint models/checkpoints/dietcorp_best.pth \
    --output simulations/dietcorp_int8.tflite \
    --quantize --representative_data data/neural_test_samples.h5
```

### 3. Run on Coral Simulator (Verilator)
```bash
python simulations/coral_npu/run_verilator_sim.py \
    --tflite simulations/dietcorp_int8.tflite \
    --input data/neural_test_samples.h5 \
    --output results/sim_latency.json
```

### 4. Analyze Results
```bash
python analysis/plot_latency_breakdown.py \
    --sim_results results/sim_latency.json \
    --output analysis/results/latency_breakdown.png

python analysis/compare_int8_accuracy.py \
    --gpu_baseline data/baseline_wer.txt \
    --sim_results results/sim_latency.json \
    --output analysis/results/accuracy_degradation.png
```

---

## Key Documents

| File | Content | Audience |
|------|---------|----------|
| **Coral_Feasibility_DietCorp_ZenBrain.md** | Full audit: op support, simulator accuracy, constraints | Researchers, architects |
| **Progress_Summary_Scope_and_Plan.md** | Investigation modes, metrics, roadmap | Team leads, advisors |
| **arch_diagram.md** | Visual architecture overview | General audience |
| **simulator_notes.md** | Verilator setup, debugging tips | Simulator engineers |
| **hardware_projection.md** | Derived specs for production hardware | Hardware partners |

---

## Known Constraints & Design Decisions

### Constraint 1: No On-Device Backprop on NPU
- **Impact**: TTA gradient steps run on host CPU driving the simulator
- **Why OK**: Mirrors production setup; host has sufficient CPU power for one Adam step per 100 ms patch
- **Design**: Simulator is orchestrated from PyTorch; forward pass routed to NPU, gradients computed in PyTorch

### Constraint 2: Static Shapes & Limited Control Flow
- **Impact**: Dynamic sequence lengths, variable beam widths not feasible on-device
- **Design**: Pre-process neural data into fixed-size patches (100 ms, 1280 features); use static beam search (width K=3)
- **Consequence**: Some flexibility lost vs. GPU, but well-matched to streaming BCI use case

### Constraint 3: Matrix Execution Unit Still Under Development
- **Impact**: Ops that don't fit the matrix spec fall back to scalar/SIMD, incurring 2–5× latency penalty
- **Mitigation**: Prioritize ops that *are* stable (matmul, softmax, ReLU); profile carefully; revisit as hardware matures
- **Timeline**: Updated simulator expected Q4 2026

### Design Decision: Simulator-as-Instrument, Not Deployment
- **Rationale**: Coral ecosystem rapidly evolving; committing to one specific chip would lock us into an unstable target
- **Approach**: Use simulation to project requirements, then evaluate production substrates (Apple, Qualcomm, Google) separately
- **Benefit**: Future-proofs the thesis against chip timeline shifts

---

## Contributing

When extending this work:

1. **Add a new simulation experiment**: Create folder under `simulations/`, add to `__init__.py`, update roadmap
2. **Implement a memory variant**: Add to `memory/`, implement `forward()` + `adapt()` interface, register in module factory
3. **New analysis**: Add plotting script to `analysis/`, reference in results section
4. **Update findings**: Revise Coral_Feasibility_DietCorp_ZenBrain.md and this README

---

## References

- **Base Architecture**: Feghhi et al., 2025. "DietCorp-Compact: Efficient test-time adaptation for neural decoding." arXiv:2507.02800
- **Memory Layer**: Bering et al., 2026. "ZenBrain: Neuroscience-grounded memory architecture for autonomous AI." arXiv:2604.23878
- **Hardware**:
  - [Google Coral NPU](https://developers.google.com/coral/): RISC-V reference IP, simulators
  - [Coral Edge TPU](https://coral.ai/): First-gen proprietary ASIC, real hardware
  - [TFLite Runtime](https://github.com/tensorflow/lite/): Inference engine
  - [Verilator](https://www.veripool.org/verilator/): Cycle-accurate RTL simulator
- **Tools**:
  - [ONNX](https://onnx.ai/): Open Neural Network Exchange format
  - [PyTorch](https://pytorch.org/): Deep learning framework
  - [TensorFlow Lite](https://www.tensorflow.org/lite): Edge inference framework

---

## Status & Next Steps

**Current Phase**: 📋 Documentation & planning (feasibility audit complete)  
**Next Immediate Step**: Implement PyTorch → TFLite conversion pipeline (Week 1)  
**Critical Path**: Baseline Characterization (Phases 1–2) by end of July 2026  
**Target Completion**: Full thesis-ready documentation by September 2026

**Questions?** Refer to:
- Feasibility audit: [Coral_Feasibility_DietCorp_ZenBrain.md](./Coral_Feasibility_DietCorp_ZenBrain.md)
- Investigation plan: [Progress_Summary_Scope_and_Plan.md](./Progress_Summary_Scope_and_Plan.md)
- Architecture: [docs/arch_diagram.md](./docs/arch_diagram.md)

---

**Last Updated**: May 28, 2026  
**Project Lead**: Pratik Pattanaik  
**Affiliation**: UC Davis Neuroprosthetics Lab
