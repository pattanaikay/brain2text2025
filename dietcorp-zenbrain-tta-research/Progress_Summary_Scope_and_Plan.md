# Progress Summary: Project Scope, Investigation Modes, and Experimentation Plan

**Prepared:** 2026-05-28 (revised after scope clarification)
**Author:** Pratik Pattanaik
**Status:** Feasibility audit complete; scope re-framed around simulation-as-instrument; entering reproduction phase

---

## 1. The Ask (what I am investigating)

### 1.1 Clinical and product motivation

The end goal is a patient-facing speech brain–computer interface (BCI) for people with ALS or brainstem stroke. Both conditions destroy motor pathways while leaving cognition largely intact, so the patient knows what they want to say but cannot say it. A BCI bypasses the broken motor pathway and decodes attempted speech directly from intracortical neural activity into text or synthesised voice.

A clinically useful system needs four properties simultaneously:

1. **Low word error rate (WER)** — text the patient and their family can actually read.
2. **Streaming, real-time decoding** — output appears as the patient is attempting to speak, not after the utterance ends.
3. **On-device feasibility on consumer hardware** — eventual deployment runs on substrate the patient already owns or can be issued (phone, tablet, laptop, small edge module), without sending raw neural data to the cloud.
4. **Robustness to distribution shift** — neural recordings drift day-to-day, so the model must adapt without re-calibration sessions.

Most published BCI papers optimise (1) at the expense of (2)–(4). My thesis is built around the four-objective version of the problem.

### 1.2 The architectural starting point

I am building on **DietCorp-Compact** (Feghhi et al., 2025, arXiv:2507.02800). It addresses all four properties in one design:

- A unidirectional causal Transformer (5 layers, hidden dim 384, 9.4 M parameters total) that decodes 256-channel microelectrode-array activity into phonemes at 10 Hz.
- Aggressive *time-masking* during training (≈53 % of each trial masked on average), which sharply delays overfitting and lifts accuracy.
- **DietCORP**, a lightweight test-time-adaptation (TTA) procedure: per trial, generate 64 time-masked augmentations, decode a pseudo-label via a beam search over n-gram LM, and take *one* AdamW step on the CTC loss — updating *only the patch-embedding module* at the front of the network.

Reported result: 12.17 % WER with a 3-gram LM (20 % relative improvement over the Brain-to-Text Benchmark '24 GRU baseline), 18 ms TTA per trial, 1.33 GiB peak GPU memory, and graceful degradation across held-out days where the no-adaptation baseline collapses from 22.7 % to 66.5 % WER over 8 days.

### 1.3 The proposed extension

I want to bolt a **multi-tier memory layer inspired by ZenBrain** (Bering, 2026, arXiv:2604.23878) on top of DietCorp's TTA pipeline. ZenBrain is a neuroscience-grounded memory architecture for autonomous AI systems: 7 layers (working, short-term, episodic, semantic, procedural, core, cross-context) + 15 algorithms (Two-Factor synaptic edges, vmPFC-coupled FSRS, Simulation-Selection sleep, neuromodulator dynamics, etc.).

The hypothesis is that giving DietCorp a layered memory hierarchy — with different time constants, prioritisation, and consolidation rules — should improve adaptation under distribution shift beyond what single-shot TTA delivers, and should do so in a biologically plausible way that future BCI research can build on.

### 1.4 What Coral actually is in this project

**Coral is an instrument, not a deployment target.** I am not planning to ship the final BCI on a Coral USB Accelerator or a Coral NPU partner SoC. Coral is being used as a simulation microscope: a representative, instrumentable edge-NPU substrate that lets me characterise how DietCorp (and a memory-augmented version of it) behave under realistic edge-deployment constraints — int8 quantization, static shapes, limited control flow, bounded on-chip memory, no on-device backprop — without committing to any specific production chip.

The eventual deployment class is "phones and laptops" (Apple Neural Engine, Qualcomm Hexagon, Google Tensor TPU, Intel AI Boost, AMD XDNA). Coral's constraint envelope is a reasonable proxy for the strictness of those platforms, and Coral is the only edge NPU with public, documented simulators today. So the simulation work serves four purposes:

1. **Characterise feasibility under realistic edge constraints** — does the model survive int8, does op compatibility hold, where does the latency budget actually go.
2. **Project hardware requirements for a future production deployment** — derive a spec sheet (MACs/sec, on-chip memory, off-chip bandwidth, op set, latency, power envelope) that any candidate production substrate can be evaluated against.
3. **Surface constraints and design issues early** — discover problems while they are still cheap to fix, before any production hardware decision is made.
4. **Produce communicable artefacts** — visualisations, latency breakdowns, and projected requirement charts that I can use in the thesis chapter and progress reports to argue *why* the design choices make sense for edge deployment.

### 1.5 The specific feasibility questions

Within that framing, the audit asked:

(a) Can DietCorp-Compact realistically be simulated on Coral — what does the model-conversion path look like, which ops are supported, what is the realistic accuracy of cycle / latency / power estimates?

(b) Can the *augmented* architecture (DietCorp + memory layer + TTA) be simulated on Coral — given Edge TPU / Coral NPU are heavily constrained?

(c) What metrics and statistics can actually be captured to make the comparison meaningful?

The audit was structured as a *deliberately skeptical* feasibility review. Full report: `Coral_Feasibility_DietCorp_ZenBrain.md`.

---

## 2. What the feasibility audit found (one-page summary, re-framed for simulation-as-instrument)

Three landing points carry the rest of the work:

**Both Coral substrates are useful, for different reasons.** The Google Coral platform is mid-transition. First-gen Edge TPU (USB Accelerator, Dev Board) is a closed Google ASIC with closed tooling but real silicon you can run today — it gives wall-clock int8 latency numbers. The new Coral NPU is an open RISC-V reference IP with two public simulators (functional MPACT-CoralNPU and cycle-accurate Verilator) — it gives cycle-level architectural visibility on an open toolchain. Since I'm not picking a deployment target, **both substrates feed the projection workstream as complementary data sources** rather than competing options.

**DietCorp's inference path is a clean fit on either substrate; the no-backprop-on-NPU limit is a non-issue under this framing.** The model is small (~10 MB int8, ~36 MFLOPs per 100 ms patch), causal, streaming, dominated by matmul/softmax/LayerNorm — comfortable for both Coral platforms. DietCORP's per-trial backward pass and Adam step run in host PyTorch driving the simulator (or the USB Accelerator); this is the DietCorp paper's existing setup with the forward path routed through the simulator, and is faithful to how a production deployment would work even on phone NPUs (where TTA gradient steps also run on the host CPU).

**ZenBrain stays largely host-side, but the simulation work can be more exploratory than a shipping-product audit would allow.** ZenBrain as published is an LLM-agent memory orchestrator — knowledge-graph storage, BM25 + embedding retrieval, FSRS scheduling, neuromodulator dynamics, reconsolidation snapshots, TripleCopy stores. Almost none of it is a tensor pipeline; ~14 of 15 components belong on the host CPU. The one on-NPU primitive that earns its place is a fixed-shape working-memory KV cache co-located with DietCorp's patch-embedding module. Under the simulation-as-instrument framing I can iterate beyond that starting point — try a 2-tier static memory, try a small graph-attention block — and let the simulator tell me where the constraints bite. Negative results ("this design breaks the 100 ms streaming budget") feed directly into the hardware-requirements projection.

---

## 3. Modes of investigation / experimentation required

The project needs seven distinct kinds of work. Listed up-front because they recur across every stage of the experiment plan.

### 3.1 Architecture and paper reading

Close reading of the DietCorp and ZenBrain papers to extract the exact op graph, parameter counts, training hyperparameters, and behavioural mechanisms. ZenBrain's appendix is where the actual algorithms live (B.1–B.11) — skipping it leads to treating ZenBrain as a neural network when it is in fact an LLM-agent memory orchestrator. Done in the audit; will be revisited per stage as the design evolves.

### 3.2 Toolchain and platform research

Documentation-level investigation of: the Coral NPU simulator stack (MPACT + Verilator), the Edge TPU compiler, the LiteRT (TFLite) converter, MLIR/IREE compilation paths, int8 post-training and quantization-aware training, and op-compatibility lists. Includes ongoing tracking of simulator maturity — for example, the Coral NPU matrix execution unit is currently "under development and evaluation" per the official docs, so cycle counts from today's Verilator sim understate the matmul throughput production silicon will deliver. Document this as an explicit uncertainty band on every projected requirement.

### 3.3 Op-level compatibility analysis

For every op in DietCorp's forward pass — patch-embed (LayerNorm/Linear/LayerNorm), the 5 Transformer blocks (LayerNorm, multi-head causal attention with T5-style relative-position bias, residual, LayerNorm, Linear, GeLU, Linear, residual), the output head — ask: does it map to a native op on the target accelerator, at what cost, and if not what does it fall back to? Determines the realistic latency floor and whether the model converts at all. Repeat after any memory-tier addition.

### 3.4 Placement reasoning (which subsystem lives where)

For each ZenBrain layer and algorithm, a structured argument for whether it belongs on the NPU, on the host CPU, or off-device. Reasoning depends on access patterns (sparse vs dense, read-heavy vs write-heavy, static-shape vs dynamic), latency budget, memory size, and whether the subsystem sits inside the streaming inference loop or runs at session boundaries. Under simulation-as-instrument framing this becomes iterative — I'm allowed to try more aggressive on-NPU placements and let the simulator push back.

### 3.5 Alternative-platform scan (as reference points, not deployment candidates)

A controlled set of comparison points so the projected hardware requirements aren't anchored to one substrate. The shortlist: first-gen Coral USB Accelerator (real int8 silicon today), Coral NPU Verilator sim (cycle-level insight on open RISC-V), and at least one reference data point from a phone-class NPU (Apple Neural Engine via Core ML is the path of least resistance). The phone-NPU run isn't a deployment commitment — it's a sanity check that the eventual deployment-class device can plausibly hit the same envelope.

### 3.6 Skeptical separation of simulator-free vs custom-instrumentation vs real-silicon-required metrics

A discipline that runs through every stage. The Coral NPU Verilator sim gives total cycles, per-instruction traces, AXI traffic, and on-chip memory residency for free. Per-op timing, streaming throughput, and the full TTA loop need custom instrumentation (CSR cycle reads, host harness scripts). Wall-clock latency on production silicon, real power, thermal effects, and end-to-end TTA latency with weight reload genuinely need real hardware — for first-gen, that's the USB Accelerator and a USB current meter; for phone NPUs, on-device profiling.

### 3.7 Hardware-requirements projection (new primary deliverable)

The newest workstream. From the simulation runs, derive a spec sheet — a concrete artefact a deployment engineer can read and act on. Targets to populate:

- **Sustained compute:** minimum MACs/sec at 10 Hz patch rate.
- **Peak compute:** instantaneous MACs/cycle for the heaviest single op.
- **On-chip memory:** activation working set plus quantized-weight residency.
- **Off-chip bandwidth:** AXI / DRAM traffic per inference, derived from simulator AXI logs and extrapolated.
- **Op set:** list of TFLite/MLIR ops the model uses; required quantization variants per op.
- **Latency budget breakdown:** per-Transformer-block contribution, per-attention-head contribution, per-LayerNorm contribution.
- **TTA cost envelope:** host-side gradient-step latency, weight-reload latency, per-trial wall-clock.
- **Energy envelope (rough):** estimated from cycle counts and a published energy-per-MAC for comparable silicon; explicitly labelled as low-confidence until validated on hardware.
- **Memory-tier delta:** for each ZenBrain-inspired memory variant, the *delta* on each of the above metrics versus vanilla DietCorp.

This spec sheet is the artefact I show people in the report. It's also what makes the thesis claim "this approach is feasible for edge deployment on consumer devices" concrete and falsifiable.

---

## 4. Planned experimentation pipeline

Four stages, each independently shippable. Objectives are framed around what simulation data we extract, not around shipping decisions.

### Stage 1 — Reproduce DietCorp-Compact baseline on toy and real data (weeks 0–2)

Goal: a known-good DietCorp setup that downstream changes can be measured against. Toy data for fast iteration; real data for headline numbers.

Work:
- Clone `github.com/ebrahimfeghhi/transformers_with_dietcorp` and reproduce 12.17 % WER with the 3-gram LM on Brain-to-Text Benchmark '24.
- Build a synthetic toy dataset that preserves the shape and temporal structure of the real input (256 features × 50 Hz bins × ~5 s trials) but uses procedurally generated phoneme sequences. Useful for quantization experiments before touching the real benchmark, and for fast simulator iteration when each sim run is slow.
- Store fp32, fp16, and dynamic-int8 reference checkpoints.

Deliverable: reproduction notebook + checkpoints + a clean toy dataset + WER metrics matching the paper within 0.5 %.

### Stage 2 — Quantization characterization (weeks 2–4)

Goal: map the int8 accuracy cliff and identify which layers are quantization-sensitive. Feeds the hardware-requirements spec.

Work:
- Export PyTorch → TFLite int8 via post-training quantization (PTQ) using ~500 validation trials as the representative dataset.
- Measure absolute WER degradation vs the fp32 baseline. Target ≤ 1 % absolute. If higher, escalate to quantization-aware training (QAT) for one pass.
- Run layer-wise sensitivity analysis: per-layer quantization error, per-layer contribution to WER. The patch-embedding module is the layer DietCORP adapts, so its sensitivity matters more than the others.

Deliverable: int8 `.tflite` model + quantization sensitivity table (per-layer error + per-layer WER contribution) + CPU latency baseline via `tflite_runtime` (the floor any NPU run must beat).

### Stage 3 — Two-substrate simulation campaign on vanilla DietCorp (weeks 4–8)

Goal: characterise DietCorp's edge-deployment envelope across both Coral substrates, and produce the first version of the hardware-requirements spec.

**Stage 3a — Coral NPU Verilator simulator.** Compile via LiteRT or MLIR to RISC-V `.elf`. Run a 10-second streaming inference with `--trace --instr_trace --debug_axi`. Extract:
- Per-Transformer-block cycle attribution.
- AXI traffic (read/write bytes) per inference, broken down by phase (patch-embed, attention QKV, attention softmax, FFN, output head).
- On-chip ITCM/DTCM residency across the run.
- Estimated steady-state latency per 100 ms patch, with explicit uncertainty band reflecting matrix-engine maturity.

**Stage 3b — Coral USB Accelerator.** Run `edgetpu_compiler` on the int8 model; capture the static op-mapping report. Benchmark via `tflite_runtime` over 1,200 trials. If LayerNorm or GeLU forces a CPU fallback, attempt fused-op rewrites and document the impact. Measure wall-clock latency, USB-bus transfer time, and (with a current meter) idle vs busy power.

Cross-substrate analysis: where do the two substrates agree and disagree on bottlenecks? Each disagreement is a finding for the report.

Deliverable: two-substrate benchmark report + Hardware Requirements Spec v1 (covering vanilla DietCorp) + a written interpretation of where the model is compute-bound vs memory-bound on each substrate.

### Stage 4 — Memory-augmented variant simulation campaign (weeks 8–12)

Goal: test the actual research hypothesis (does a neuroscience-inspired memory layer improve TTA under distribution shift?) and produce Hardware Requirements Spec v2 capturing the deltas the memory addition introduces.

Work:
- **Variant A (conservative starting point):** add a static-shape KV cache of size N = 8 holding recent patch embeddings, plus a learnable cross-attention gate before the first Transformer block. Re-quantize, re-simulate on both substrates.
- **Variant B (more ambitious, if variant A leaves headroom):** add a second static-shape tier with longer effective horizon (e.g. N = 16 with a learned decay scalar mimicking a TripleCopy fast/medium pair). Re-quantize, re-simulate.
- **Variant C (stress test):** any simulation result that breaks the 100 ms streaming budget is *kept* and reported as a constraint finding, not discarded — it directly informs the spec sheet ("this variant requires ≥ Xx more compute than the substrate provides").
- Implement the host-side ZenBrain-inspired components that influence between-trial behaviour: a working-memory tier (the static KV cache), a short-term-memory buffer (session-bounded), a simple decay scheduler (single Ebbinghaus-style τ at first; TripleCopy with three τ values if time permits), and a priority signal that gates which trials feed the TTA gradient step.
- Drive the full DietCORP TTA loop from host PyTorch: 64 augmented forwards through the simulator/USB Accelerator, beam-search decode + pseudo-label on host, gradient step on patch-embed + gate on host, weight reload to the device. Measure per-trial wall-clock budget.
- Run the 5-held-out-day and 8-held-out-day evaluation from the DietCorp paper. Compare each memory-augmented variant against vanilla DietCORP.

Bonus: replicate the best-performing variant on Apple Neural Engine via Core ML as a phone-NPU sanity check.

Deliverables: WER degradation curves under distribution shift (vanilla vs each variant) + latency overhead breakdown for each variant + Hardware Requirements Spec v2 with per-variant deltas + a written analysis of which ZenBrain mechanisms earned their place inside the inference graph, which sit usefully around it, and which add cost without benefit.

---

## 5. Tools and substrates I will need

Consolidated list so procurement / setup can happen up-front.

| Category | Item | Stage | Notes |
|---|---|---|---|
| Hardware (workstation) | RTX-3090 or equivalent | 1, 2, 4 | DietCorp training, host-side TTA gradient step, simulator driving |
| Hardware (edge, instrument) | Coral USB Accelerator (~$60) | 3b, 4 | Cheapest real int8 silicon for wall-clock numbers |
| Hardware (edge, reference) | iPhone or M-series Mac for Apple Neural Engine | 4 bonus | Phone-NPU sanity check; supports "phones/laptops" deployment claim |
| Software | PyTorch + accelerate + standard training stack | 1, 2, 4 | Existing toolchain |
| Software | TFLite + LiteRT + edgetpu_compiler | 2, 3b, 4 | Quantization + Edge TPU compile |
| Software | Coral NPU repo (MPACT-CoralNPU + Verilator sim) | 3a, 4 | Bazel build; Debian Trixie best-tested host environment |
| Software | Core ML Tools (`coremltools`) | 4 bonus | ANE deployment |
| Software | Plotting + analysis (matplotlib, seaborn, polars) | 3, 4 | For the visualisations the report needs |
| Datasets | Brain-to-Text Benchmark '24 (public) | 1, 2, 4 | Headline numbers |
| Datasets | Synthetic toy dataset (procedurally generated) | 1, 2, 3a | Fast iteration; simulator runs |
| Compute | One GPU-week for QAT fallback in Stage 2 | 2 | Only if PTQ degradation exceeds 1 % |
| Instrumentation | USB current monitor (~$30) | 3b, 4 | Real power numbers on the USB Accelerator |

---

## 6. Risks and open decisions

Four pending decisions. Re-framed for the simulation-as-instrument scope.

1. **Which Coral substrate gets *primary* simulator attention?** Both are needed, but compute budget may force prioritisation. First-gen USB Accelerator turns around faster and gives wall-clock numbers directly. Coral NPU Verilator gives the cycle-level architectural detail that makes the Hardware Requirements Spec rigorous. If forced to pick: Verilator first, USB second, because the spec sheet is the primary deliverable.

2. **How aggressively do I escalate memory-tier variants in Stage 4?** The conservative Variant A is safe and likely fits the streaming budget. Variant B and Variant C are more ambitious and may produce useful negative results, but they cost simulation time. Initial plan: A first, B if A leaves >20 % of the streaming budget unused, C as a stress-test ablation.

3. **Toy dataset fidelity.** A procedurally generated dataset for fast iteration is appealing, but it has to be faithful enough that quantization and latency results transfer to the real benchmark. The safe path is: generate the toy data by perturbing recorded calibration-day trials from the public benchmark (preserves real noise statistics); use a fully synthetic generative model only for early simulator-loop debugging.

4. **How thoroughly should the phone-NPU bonus run be pursued?** Light-touch (one model, one device, single inference timing) is enough to support the deployment-class claim. Deep characterisation matches the Coral work but doubles the bonus-stage effort. Initial plan: light-touch in the thesis chapter, with deep characterisation reserved for follow-on work if reviewers ask.

---

## 7. What I'd ask reviewers / advisors to weigh in on

Three specific decisions where outside input would change the plan:

- Whether to invest in QAT up-front in Stage 2 or wait to see PTQ results. QAT roughly doubles training cost but is the standard way to recover 1–2 % of WER on Transformer quantization.
- Whether the Hardware Requirements Spec deliverable is more useful as a single combined document or as per-variant spec sheets (vanilla DietCorp, Variant A, Variant B). The single document is tidier; per-variant sheets are easier for a deployment engineer to read.
- Whether the negative-result framing in Stage 4 is acceptable — the plan explicitly treats "memory variant X breaks the streaming budget" as a publishable input to the spec sheet, but some advisors prefer all-positive narratives.

---

## 8. Visual summary of the project shape

A loose flow diagram of where the work sits, for the report's intro slide.

```
                  Patient (ALS / brainstem stroke)
                              │
                       intracortical MEA
                              │
                    256-channel neural activity
                              │
            ┌─────────────────┴─────────────────┐
            │     DietCorp-Compact (forward)    │  ← runs through the
            │  patch_embed → 5× Transformer →   │     Coral simulator
            │           output_head             │     (instrument, not target)
            └─────────────────┬─────────────────┘
                              │
                         phoneme logits
                              │
                  beam search + n-gram LM        ← host CPU
                              │
                              ▼
                         decoded text
                              │
            ┌─────────────────┴─────────────────┐
            │   DietCORP TTA (between trials)   │  ← gradient step on host CPU,
            │  pseudo-label + 64 aug forwards   │     forwards through the simulator,
            │   → one Adam step on patch_embed  │     weights reloaded to the device
            └─────────────────┬─────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────┐
            │   ZenBrain-inspired memory layer    │  ← mostly host-side;
            │  working / short-term / episodic /  │     working-memory tier
            │  semantic / procedural / core /     │     fused into the simulated
            │            cross-context            │     inference graph
            └─────────────────┬───────────────────┘
                              │
                              ▼
                  ┌───────────────────────────┐
                  │  Hardware Requirements    │  ← the primary report artefact:
                  │       Spec (v1, v2)       │     MACs/s, on-chip memory, ops,
                  │                           │     latency, bandwidth, energy,
                  │                           │     per-variant deltas
                  └───────────────────────────┘
                              │
                              ▼
            (informs future deployment-substrate
              choice — phone NPU, laptop NPU,
              dedicated edge module, …)
```

The Coral simulator sits at the centre of the diagram as a *microscope*, not a destination. The downstream deployment substrate is deliberately drawn as a future, undecided element.

**Analogy.** Coral is to this project what a wind tunnel is to an airframe designer — you don't fly a plane in a wind tunnel, but the data you collect there tells you what configuration will fly when it leaves the building. The Hardware Requirements Spec is the equivalent of a flight envelope: a falsifiable specification that any candidate deployment substrate can be matched against.

---

## 9. Status

- Feasibility audit complete (28 May 2026). Output: `Coral_Feasibility_DietCorp_ZenBrain.md`. Scope re-framed (Coral as instrument, not deployment target); this document supersedes the original plan.
- Stage 1 reproduction work begins next week.
- Stage 3a Verilator sim build can begin in parallel using a stub MobileNet, so the simulator pipeline is validated before DietCorp is ready to deploy.
- Decisions in Section 6 to be resolved with advisor in the next 1:1.
