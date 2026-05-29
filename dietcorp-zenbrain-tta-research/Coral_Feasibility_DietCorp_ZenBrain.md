# Feasibility Assessment: Simulating DietCorp + a ZenBrain-style Memory Layer on Coral

**Author context (per request brief and follow-up Slack post):** the long-term goal is patient-facing assistive tech for ALS and brainstem-stroke speech BCIs. The candidate base architecture is *DietCorp-Compact* — the time-masked Transformer + DietCORP TTA from Feghhi et al. (Brain-to-Text Benchmark '24, arXiv:2507.02800v2). The proposed extension is a memory hierarchy inspired by Bering's *ZenBrain* (arXiv:2604.23878v1). The deployment target framed in the Slack post is "open-source quantized models on laptops/phones"; Coral is being treated as a proxy for studying int8 edge inference.

**Date prepared:** 2026-05-28.

**Note on attachments.** The brief said a thesis was attached but only the two papers came through. The user's later Slack-post excerpt now stands in for that context, so no re-attachment is required — but if there is a fuller thesis draft (architecture diagram, training set details, target hardware shortlist), sending it would let me tighten Sections 3 and 7. No image came through in the follow-up either; I worked from the transcribed text alone.

---

## 0. Executive summary (TL;DR)

1. **There are two Corals, and the question changes depending on which one you mean.** The Google Coral platform is mid-transition. The first-generation **Edge TPU** (USB Accelerator, Dev Board, Mini) is a closed Google ASIC with a closed Edge TPU Compiler and **no public cycle-accurate simulator**. The new **Coral NPU** (whose simulator page you linked) is a RISC-V open reference IP with two public simulators: MPACT-CoralNPU (functional, fast) and `rvv_core_mini_axi_sim` (Verilator, cycle-accurate, slow). DietCorp-Compact is a viable workload for both, but the toolchains, op support, simulator semantics, and what counts as "real" deployment hardware are all different. Most published Coral practice is still first-gen; the new NPU is open and promising but its **matrix execution unit is explicitly listed as still under development as of Jan 2026**, so cycle-accurate numbers from the new sim today underestimate matmul throughput compared to silicon.

2. **(a) DietCorp-Compact is simulatable, with caveats.** It is small (9.4 M params, 364 MFLOPS, ~10 MB int8), causal, streaming, and dominated by matmul/softmax/layernorm — a clean fit on paper. The hard part is not inference but *DietCORP itself*: TTA requires a backward pass and one Adam step per trial, and **neither the first-gen Edge TPU nor the Coral NPU runtime supports on-device training**. The clean separation is: forward pass (inference) on the NPU, gradient computation and parameter update on the host CPU. For the new Coral NPU, the Verilator simulator gives genuinely cycle-accurate inference latency; power and energy numbers do **not** come from the simulator and require either a synthesis flow or real silicon. For the first-gen Edge TPU, "simulation" is largely the compiler's static compatibility report plus a CPU-reference TFLite run — there is no cycle-accurate behavioral model in public release.

3. **(b) Bolting ZenBrain on top is mostly *not* an NPU problem.** ZenBrain as published is an LLM-agent memory architecture — a coordinator over knowledge-graph storage, BM25+embedding retrieval, FSRS schedulers, a four-channel neuromodulator with phasic/tonic decay, reconsolidation snapshots, TripleCopy stores with three τ values, and so on. Almost none of it is a tensor pipeline; it is database, scheduling, and graph code. **At least 80 % of the ZenBrain stack should live on the host CPU, not on the NPU.** The single piece with a defensible NPU placement is the embedding/similarity used for `recall(query)` — and even that is only on the NPU if you co-locate it with DietCorp's inference graph as a fused encoder. The honest message is: simulating "DietCorp + ZenBrain on Coral" is really "simulating DietCorp on Coral while ZenBrain runs as ordinary Python/C++ on the host." A tractable subset that *is* worth simulating is the **working-memory / fast-decay tier as a static fixed-shape KV cache fused with the patch-embedding module** — that is on-NPU, that is what changes inference latency, that is what TTA actually adapts.

4. **(c) Useful simulator metrics are narrower than they look.** The Verilator sim gives cycle counts, per-op cycle breakdowns, AXI memory transactions, and instruction traces — that is enough for inference latency, per-layer timing, on-chip vs off-chip traffic, and crude memory-bandwidth pressure. It does **not** give wall-clock latency on real silicon, power, dynamic energy, thermal effects, or DRAM contention with other SoC workloads. Accuracy degradation from int8 quantization is captured offline at conversion time (PyTorch → TFLite int8 → compile), not by the simulator. TTA convergence on-device is fundamentally not a simulator question; it is an end-to-end software experiment that needs whatever host pipeline you build around the NPU. Realistic accuracy: the cycle-accurate sim is within a few percent on inference latency for what the matrix unit *does* support today; for ops that fall back to scalar/SIMD-only execution it can mislead by 2–5× until the matrix engine matures.

5. **Recommendation.** For the thesis claim "this stack can run on-device for ALS patients," **do not anchor on the Coral simulator as the centrepiece**. Use it as one of three data points: (i) Coral NPU Verilator sim for cycle-accurate inference timing and per-op profiling of DietCorp-Compact's forward pass; (ii) first-gen Coral USB Accelerator or Dev Board Mini ($60–$150) for actual int8 wall-clock measurement; (iii) a phone/laptop NPU run (TFLite + XNNPACK on Pixel Tensor / Apple Neural Engine via Core ML / Qualcomm QNN) so the claim generalises to the deployment target the Slack post actually names. The minimum-viable experiment in §8 is structured around that triangulation.

---

## 1. The two Corals: a clarifying paragraph before everything else

The brief refers to "the Coral simulator (https://developers.google.com/coral/guides/software/simulator)" and to "model conversion (PyTorch → TFLite → Edge TPU compiler)" in the same sentence. Those describe different platforms and the distinction matters.

| | **First-gen Coral (Edge TPU)** | **Coral NPU (the new platform)** |
|---|---|---|
| Released | 2019 | 2025–2026, partner silicon rolling out |
| Silicon | Google proprietary Edge TPU ASIC | Open RISC-V reference IP (RV32IMFV / Zve32x), partner-manufactured (e.g. Synaptics Torq) |
| Compiler | Closed Edge TPU Compiler (`edgetpu_compiler`) on top of TFLite | LiteRT (TFLite) converter **or** MLIR/IREE; both open |
| Datatype | int8 / uint8 fully quantized | int8 / int16 / int32 native |
| Simulator | None public; closest is CPU-reference TFLite run + compiler static report | **MPACT-CoralNPU** (functional, fast) and **`rvv_core_mini_axi_sim`** (Verilator, cycle-accurate) |
| Power | 2 W typical, 0.5 TOPS int8 | Targets hundreds of GOPS today, scalable to multi-TOPS |
| What's mature | Compiler, runtime, hardware all stable | Scalar core stable; vector unit stable; **matrix (MAC) execution unit "under development and evaluation" per the ML-inferencing arch doc, last updated 2026-01-21** |
| Best for | Wall-clock measurement on real silicon, today | Cycle-level architectural study, hardware co-design, MLIR-toolchain work |

The simulator page you linked is the Coral NPU one. So the simulator question in (a) and (b) is really a Coral NPU question. The "PyTorch → TFLite → Edge TPU compiler" workflow you described is the first-gen one. Both are legitimate paths and the thesis should be explicit about which it's chasing in each experiment.

**Visual analogy.** First-gen Edge TPU is like a sealed appliance with a printed performance spec — you plug your model in and measure what happens. Coral NPU is like a chip-design sandbox shipped as RTL — you can step through every cycle and inspect every register, but the appliance hasn't shipped yet. Most "is it on-device?" questions for the thesis want the appliance answer. Most "why is it slow?" questions want the sandbox.

---

## 2. DietCorp-Compact as a workload — what we're trying to simulate

From the paper (§3.5, Appendix D, Table 6):

- **Architecture:** unidirectional causal Transformer, 5 blocks, hidden dim 384, 6 heads, dim/head 64, FFN multiplier 4, GeLU activation. Pre-LayerNorm. Causal attention mask. T5-style learned-scalar relative positional bias added to attention logits.
- **Input pipeline:** 256 neural features × 5 time bins per patch (100 ms each, non-overlapping). Patch embed = LayerNorm → Linear (256·5 = 1280 → 384) → LayerNorm.
- **Output pipeline:** final LayerNorm → Linear → 42-class logits (40 phonemes + CTC blank + silence). Emitted every 100 ms (10 Hz).
- **Size and cost:** 9.4 M parameters; 364 MFLOPS (Table 2); peak GPU memory 2.66 GiB at fp32 training. Int8 inference weight footprint ≈ 9.4 MB.
- **DietCORP TTA:** for each held-out trial, generate Z = 64 augmented copies via white-noise / baseline-shift / time-masking, decode logits to a pseudo-label with 3-gram LM + beam search, then take **one** AdamW gradient step on the CTC loss against the pseudo-label, updating **only the patch-embedding module** (the LayerNorm→Linear→LayerNorm at the front). The 5 Transformer blocks and the output head stay frozen. Reported as 18 ms per trial and 1.33 GiB peak GPU memory on an RTX 3090.
- **Streaming budget:** real-time means each 100 ms patch's forward pass must finish in well under 100 ms; the trial-level TTA step happens between trials, not in the streaming loop.

This is a *small* model by NPU standards. The Edge TPU's typical workload is MobileNet-class CNNs at ~5 MB; the new Coral NPU is sized for "ambient sensing" wearables. DietCorp fits comfortably either way as long as the ops are supported.

**Critical observation about ops.** A Transformer attention block has: (1) matmul Q = xW_q, K = xW_k, V = xW_v; (2) scaled dot product QK^T / √d; (3) **scalar bias add** for the T5 relative position; (4) **causal mask add** (a large negative constant added at masked positions); (5) softmax; (6) matmul softmax · V; (7) output proj. Plus LayerNorm and GeLU surrounding all of it. The interesting ones for compatibility are softmax, layernorm, and GeLU.

---

## 3. Part (a) — Can DietCorp-Compact realistically be simulated on Coral?

### 3.1 The conversion path

The clean PyTorch → silicon-ready path on both Coral generations is:

```
PyTorch model
    │  (1) tracing / scripting / export
    ▼
ONNX                                                     [optional but useful]
    │  (2) onnx-tf or direct torch.export → TFLite       
    ▼
TFLite float graph
    │  (3) post-training int8 quantization, with
    │      a representative dataset of neural-data samples
    ▼
TFLite int8 graph (.tflite)
    │  (4a) Edge TPU Compiler          (4b) LiteRT/IREE for Coral NPU
    ▼                                  ▼
edgetpu .tflite                        .elf / .bin via MLIR or LiteRT
    │                                  │
    ▼                                  ▼
USB Accelerator / Dev Board            MPACT-CoralNPU or Verilator sim
```

Three places where this fails or needs surgery for a Transformer:

1. **Quantization-aware ops.** Post-training int8 quantization with a representative dataset (a few hundred trial windows from validation) gives the best chance of preserving accuracy. Quantization-aware training (QAT) helps further but doubles training cost. **Expected accuracy hit from int8: 0–2 % absolute WER**, based on what others have seen converting similar-size Transformers; LayerNorm and softmax are the most-quantization-sensitive ops, and DietCorp uses LayerNorm aggressively (pre-attention, pre-FFN, in the patch embed). Plan to budget for QAT if PTQ comes in worse than 1 % WER degradation.

2. **Ops that don't map cleanly on first-gen Edge TPU.** The first-gen Edge TPU's supported-op list is centred on CNN primitives — CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, MAX_POOL, SOFTMAX, RESHAPE, ADD, MUL, LOGISTIC, QUANTIZE/DEQUANTIZE. Modern compiler versions added SLICE, STRIDED_SLICE, BATCH_MATMUL, and TRANSPOSE, but the support is conditional (must be int8, must have static shapes, dimensions must be within size constraints). Three DietCorp ops are at risk:
   - **LayerNorm.** Not a native Edge TPU op for older compiler versions. If unfused, decomposes into mean/variance/subtract/multiply/add — some of those run on the TPU, some fall back. The first unsupported op typically forces *everything after it* back to CPU, which kills throughput. The fix is to fuse LayerNorm into the matmul before it via the converter, or to use a custom op.
   - **GeLU.** Same story — GELU is in newer TFLite, but Edge TPU compiler version determines whether it executes on-TPU. If unsupported, falls back to CPU.
   - **The T5 relative-position bias.** This is a learned vector indexed by relative distance and added to attention scores. It's a gather + add. Gather/embedding-lookup behaviour on Edge TPU is finicky; absolute sinusoidal embeddings are easier to map but Table 4 of the DietCorp paper shows that removing T5 positions costs ~10 % WER. **Don't blindly swap embeddings to make compilation easier.**

   The Coral NPU path is more forgiving because MLIR/IREE compile down to RISC-V scalar + RVV vector instructions, and softmax/LayerNorm/GeLU are computed in software using the vector unit. There is no "supported op list" gate in the same sense — anything you can express in MLIR you can lower. The trade-off is that ops not landing on the matrix engine run on the vector engine at lower throughput; on the current sim with the matrix engine still under development, attention matmul will mostly land on RVV VDOT instructions, which will look slower than it will be on production silicon.

3. **DietCORP is training; on-device inference platforms don't train.** This is the biggest single issue and the brief asked for honesty about it. From the LiteRT page (last updated 2026-02-12): *"LiteRT itself is primarily an inference framework and does not support model training."* The first-gen Edge TPU has never supported on-device backprop. Even Coral NPU as currently documented is an inference target.

   What this means concretely: the forward passes for the 64 augmented copies of each trial can run on the NPU, but the loss computation, backward pass, optimizer state, and weight update for the patch-embedding module have to run on the host CPU. So a faithful DietCORP loop is:

   ```
   per trial:
       on-NPU:   64 × forward(augmented_x)  →  64 × logits
       on-CPU:   beam-search w/ 3-gram LM   →  pseudo-label
       on-CPU:   forward + backward through patch_embed (small!)
                 + Adam step on patch_embed weights
       on-NPU:   reload updated patch_embed weights into the int8 model
   ```

   The "reload updated weights" step matters: the standard Edge TPU runtime caches compiled weights on the device. Hot-reloading per-trial-updated weights with re-quantization is feasible but adds latency. Plan for ~5–20 ms of conversion+reload overhead per trial in addition to the gradient step, on top of the paper's RTX-3090 18 ms baseline.

### 3.2 What the simulator emulates vs. what only runs on real hardware

**For the first-gen Edge TPU there is no behavioural simulator in public release.** What you have is:
- The Edge TPU **Compiler** itself, which produces a static report: ops mapped to TPU, ops falling back to CPU, on-chip data cache utilisation estimate, and a *very* rough projected speedup.
- A CPU-side TFLite reference run via the standard TFLite interpreter for functional correctness (does the int8 model produce the same outputs as the float model, modulo quantization noise).
- The real USB Accelerator or Dev Board, run with `tflite_runtime` and the libedgetpu delegate, for actual timing. This is the only path that gives real numbers.

**For the new Coral NPU you have two genuine simulators:**

- **MPACT-CoralNPU** (`bazel build //sim:coralnpu_v2_sim`). Functional / instruction-level. Fast. Confirms the compiled `.elf` runs end-to-end and produces correct outputs; supports interactive debugging (break, step, reg get/set, mem get/set). Default memory map: ITCM 8 KB at 0x0000, DTCM 32 KB at 0x10000. **Does not give cycle counts.**
- **`rvv_core_mini_axi_sim`** (Verilator). Cycle-accurate. Slow (Verilator generally runs 10⁴–10⁶× slower than real). Produces VCD traces, AXI traffic logs, and per-instruction cycle accounting. Run with `--trace --debug_axi --instr_trace` to get the full data.

For DietCorp-Compact a useful sanity check: at 364 MFLOPS averaged over 1 s of input and a 10 Hz output rate, the cost is ~36 MFLOPs per 100 ms patch. The Coral NPU matrix engine targets 256 MACs/cycle at int8. At a representative 400 MHz clock, that's ~100 GMACs/s peak, which is ~3 orders of magnitude headroom. The realistic streaming inference latency on production silicon should land in the single-digit milliseconds. **The Verilator sim today, with the matrix engine still under development, will probably report this number 2–5× higher than the silicon will deliver**, because attention matmul is hitting the vector path instead of a dedicated MAC array. Calibrate any "DietCorp meets real-time on Coral" claim with that uncertainty band.

### 3.3 Realistic accuracy of cycle / latency / power estimates

| Metric | Coral NPU sim accuracy | First-gen Edge TPU |
|---|---|---|
| Inference latency (cycle count) | Verilator: cycle-accurate vs. RTL, but RTL ≠ partner silicon. Expect ±10 % vs. the partner SoC, larger if the matrix engine is still being parameterised | Compiler estimate: ±2× honestly. Real hardware: actual measurement, no estimate needed |
| Per-op timing | Verilator with instruction trace: very good for what's expressed in RISC-V/RVV; misleading for ops that *should* hit the matrix engine but currently run on RVV | Compiler report shows ops on/off TPU; per-op timing must come from hardware via `tflite_runtime` profiler |
| On-chip memory utilisation | Yes, can be read from AXI traffic — good fidelity | Compiler report gives on-chip data cache used (out of 8 MB), but allocation policy is opaque |
| Memory bandwidth pressure | Yes, AXI traffic logs give read/write bytes per cycle | Indirect: high cache miss → high host-side latency |
| Power / energy | **No.** Verilator simulates logic, not transistors. Requires synthesis with a power-estimation tool (e.g. Synopsys PrimeTime) | **No.** Need a USB current monitor or PCIe power telemetry on the dev board |
| Quantization accuracy | **Offline.** Computed at TFLite conversion time, not by the simulator | Offline same way |
| TTA convergence behaviour | **Not a simulator question.** Needs end-to-end software run | Same |

**Bottom line for (a):** DietCorp-Compact inference is well within both Coral platforms' envelopes. The simulator gives credible cycle and latency numbers (Coral NPU only) and credible op-compatibility verdicts (Edge TPU compiler). The simulator gives essentially nothing on power, on TTA dynamics, or on quantization accuracy — those need other instruments.

---

## 4. Part (b) — DietCorp + a ZenBrain-style memory layer + TTA on Coral

### 4.1 The framing problem: ZenBrain is not what NPUs are for

The brief asked me to think hard about this and I will be direct: ZenBrain as described in arXiv:2604.23878 is **an LLM-agent memory architecture, not a neural-network internal memory layer**. It is a Python/TypeScript-level orchestrator that sits on top of an LLM (Claude 3.5 Sonnet is the backbone in the paper). Reading the appendix carefully (B.1–B.11, C, Table 15, I):

- The 7 layers are *stores of structured facts and embedded text*, not tensor activations. Working memory holds ~7 items. Episodic memory is timestamped events. Semantic memory is a knowledge graph with edges carrying weight w_ij and a Fisher-information proxy. Procedural memory is tool-use patterns. Cross-context memory is entity resolution across domains. These are records and graph edges, not vectors flowing through layers of a network.
- The MemoryCoordinator's five operations (`store`, `recall`, `consolidate`, `decay`, `review`) are **database operations augmented with LLM-mediated reasoning** — `consolidate()` calls an LLM to abstract episodic into semantic, `review()` schedules FSRS reviews, etc.
- The 15 algorithms are statistical / scheduling / control logic: Two-Factor synaptic edges are an EWC-style penalty over KG weights; vmPFC-coupled FSRS is an interval scheduler; Simulation-Selection sleep is a CA3/CA1-inspired offline RL replay loop; Bayesian confidence is Bayes' rule on fact probabilities; the four-channel NeuromodulatorEngine maintains four scalar parameters with 5-min-half-life phasic bursts; Reconsolidation is PE-gated with four update modes; TripleCopyMemory stores three copies with three decay constants τ_f=4h, τ_m=14d, τ_d=7d; PriorityMap is a 4-D weighted sum.
- The retrieval primitive is BM25 + cosine similarity over `nomic-embed-text` (768-d) — that's the only piece of ZenBrain that looks like a tensor pipeline.

The neuroscience inspiration is real; the implementation is fundamentally a key-value store + KG + scheduler + LLM calls. Telling an Edge TPU or a Coral NPU to run that is a category error — those are quantized matmul accelerators with static shapes, no dynamic allocation, no hash maps, no OS, no interrupts, and (for Coral NPU) "executor run-to-completion model" with bare-metal programs.

### 4.2 Placement: where each ZenBrain tier should live

The table the brief asked for. "TPU" here means whichever Coral NPU/Edge TPU is in scope; "CPU" means the host SoC's main core; "off-device" means a server, cloud, or workstation that the BCI talks to.

| ZenBrain tier / component | Access pattern | Where it belongs | Reasoning |
|---|---|---|---|
| **Working memory** (~7 active items) | Tiny, fast, frequently rewritten | CPU (and possibly fused into NPU's patch-embed state) | Small enough to live in a static fixed-shape buffer; if you want NPU acceleration, model it as a fixed-N KV cache concatenated to DietCorp's patch tokens |
| **Short-term memory** (session context) | Variable size, session-lived | CPU | Variable shapes; needs to grow within session |
| **Episodic memory** (timestamped events) | Append-mostly, queried by time/embedding | CPU + off-device | Timestamps, JSON-ish records, knowledge-graph integration — none of which is matmul |
| **Semantic memory** (KG with Two-Factor edges) | Graph traversal, edge-weight updates | CPU + off-device | KG storage is a database; edge updates are EWC-style penalties computed on the host |
| **Procedural memory** (tool/workflow patterns) | Read-heavy, occasional writes | CPU | Not relevant to a BCI; carry-over from the agent setting |
| **Core memory** (pinned identity facts) | Always-resident, never decays | CPU (and possibly NPU as fused constants) | If you're decoding personal/biographical content, you could fuse a small fact vocabulary into the LM at the beam-search stage |
| **Cross-context memory** | Cross-domain entity resolution | CPU + off-device | Same as semantic |
| `recall()` — BM25 + cosine fusion | Sparse + dense similarity | **Dense part on NPU**, BM25 on CPU | The embedding similarity is a matmul; if you co-locate the encoder on-NPU it's the one ZenBrain primitive that earns its place there |
| `consolidate()` — episodic → semantic via LLM | LLM call | Off-device (or large host LLM) | LLM not running on the BCI's NPU |
| `decay()` — Ebbinghaus over fact strengths | Scalar arithmetic over records | CPU | Tiny |
| FSRS scheduler + vmPFC coupling | Tabular | CPU | Tiny |
| NeuromodulatorEngine (4 scalars × phasic/tonic) | ODE-like state | CPU | Tiny; these are *hyperparameters* the rest of the stack reads |
| ReconsolidationEngine (PE-gated, snapshot logging) | Disk writes, KG mutations | CPU + persistent storage | Snapshot logging is filesystem I/O |
| TripleCopyMemory (3 τ values, max() composite) | Three decay computations per record | CPU | Trivial; ~ns per record |
| PriorityMap (4-D weighted sum, dynamic weights) | Tiny dot product per access | CPU | Could live on NPU but the cost of the round-trip dominates the savings |
| StabilityProtector, MetacognitiveMonitor | Audit/gating logic | CPU | Not numerical |

**The honest summary:** of the 15 ZenBrain algorithms plus 6 PMA components, **one** (the embedding similarity inside `recall`) has a real argument for living on the NPU, and even there the argument is that it's already on the NPU as part of DietCorp's encoder, not that you would put it there for ZenBrain's sake.

### 4.3 The tractable subset to actually simulate

The brief explicitly asked: if (b) is largely infeasible because Edge TPU can't handle the dynamic memory patterns ZenBrain implies, say so and propose what a tractable subset would look like. So: yes, it is largely infeasible to put ZenBrain on the NPU, and the tractable subset is the following — call it **"DietCorp + a single-tier on-NPU fast memory":**

1. **A fixed-shape on-NPU KV cache** of size N (say N = 8 or 16) holding the most recent patch embeddings. This is the part of "working memory" that you can express as a static tensor.
2. **A learnable scalar gate** that lets the current patch attend over the cache before going into the existing Transformer. This is a tiny add: one extra LayerNorm + one cross-attention block, both static-shape, both int8-quantizable, both Edge-TPU-mappable.
3. The other six ZenBrain tiers and all 14 of the other algorithms live on the host CPU and **never enter the NPU graph**. They influence the system at session boundaries, between trials, and during overnight consolidation — not inside the streaming inference loop.
4. **DietCORP's TTA still updates only the patch-embedding module**, exactly as in the original paper, plus optionally the new gate scalar. Patch embed remains tiny, so the host-side gradient step stays at ~tens of ms.

This subset gives you a real, simulatable change to the inference graph that you can A/B against vanilla DietCorp on the Coral NPU sim. It also gives you a defensible thesis claim: *"we added a neuroscience-inspired working-memory tier and showed it improves WER under distribution shift without breaking the streaming inference budget on a representative edge NPU."* That is much stronger than "we put a 7-tier memory hierarchy on a Coral chip" — which would not be true.

### 4.4 Hybrid CPU + NPU simulation: is it supported?

Half-yes. The Coral NPU Verilator simulator is an NPU-only simulator — it does not simulate a host CPU running Linux alongside it. The intended setup for the NPU when it ships is exactly hybrid: the partner SoC's main CPU drives the NPU via the AXI interface, and the NPU runs bare-metal `.elf` programs invoked per inference. For *simulating* the hybrid system you have three options:

1. **Run host-side ZenBrain in normal Python on your workstation, and the NPU portion in the Verilator sim**, with a custom harness that pipes inputs into the sim and reads outputs back. Wall-clock divergence between the two halves is meaningless; you measure NPU-side cycle counts and host-side software timing separately.
2. **Use Renode** (Antmicro) to co-simulate a host CPU + the Coral NPU as a peripheral. Renode supports RISC-V cores and is the natural choice for "I want a whole SoC, not just one IP block." It will not give you cycle accuracy for either part, but it gives you a coherent event timeline across the boundary.
3. **For first-gen Edge TPU**, hybrid CPU + accelerator simulation reduces to: run the host Python code normally, and treat the Edge TPU as an opaque library called from that code. Timing comes from either real hardware (USB Accelerator) or the compiler's static estimate.

I would not recommend gem5 here. Gem5 is excellent for CPU architecture research but does not model the Coral NPU and would require you to write your own NPU module — multi-person-month work.

---

## 5. Part (c) — What metrics can actually be captured

The brief asked for the distinction between (i) what the simulator gives you for free, (ii) what needs custom instrumentation, and (iii) what genuinely requires the dev board / real hardware.

### 5.1 Coral NPU Verilator simulator — what it gives for free

- **Total inference latency in cycles**, from `--cycles` and the simulation completion timestamp.
- **VCD waveform trace** (`--trace`) — every signal toggle. Useful for verifying expected stall patterns, but at the volume Verilator produces, viewable only on small program slices.
- **Instruction trace** (`--instr_trace`) — every RISC-V instruction executed and its PC. Combined with the `.elf` symbol table, this gives **per-function and per-op cycle attribution**. For DietCorp, you get per-block timing: patch embed cycles, per-layer-of-Transformer cycles, output head cycles, softmax cycles.
- **AXI traffic log** (`--debug_axi`) — every memory transaction, address, byte count, direction. Gives you **on-chip ↔ off-chip memory bandwidth and pressure**, and lets you spot weight reloads and activation spills.
- **On-chip memory utilisation** — from how much of ITCM/DTCM the linker placed live regions in; visible from the `.elf` segments and verifiable in the sim.

### 5.2 Coral NPU — what needs custom instrumentation

- **Per-op timing for ops you care about specifically.** Wrap each op call in a `csrrs` cycle-counter read pair (RISC-V counters), or insert `ebreak` markers around regions and post-process the cycle counts between markers.
- **Streaming throughput.** Run inference repeatedly for, say, 10 seconds of simulated input, and divide. This requires a test harness that feeds 100 ms patches sequentially through the same NPU program with state preserved.
- **DietCORP TTA convergence.** Build a host-side Python loop that drives the sim. Per trial: run 64 augmented forward passes via the sim, decode pseudo-label, run the host-side gradient step in PyTorch, push updated weights back to the sim's memory map, repeat. Measure WER on held-out days as in the paper. The sim is incidental here; this is a software experiment.
- **Quantization accuracy degradation.** Offline. Convert the model with PTQ, evaluate on validation set, compare WER vs the float baseline. Do separately for the patch-embed quantization sensitivity since that's the part DietCORP updates.

### 5.3 What genuinely requires real hardware

- **Wall-clock latency on the actual deployment device.** No simulator gives this for production silicon; you need either a Coral partner-SoC dev board (when those ship — Synaptics Torq, etc.) or the existing first-gen Coral USB Accelerator / Dev Board Mini for an apples-to-oranges proxy.
- **Power and energy.** The simulator does not model transistor switching. Real measurement = USB current meter on the Accelerator, or platform-specific power telemetry on the Dev Board.
- **Thermal effects.** Throttling under sustained load only shows up in silicon.
- **End-to-end TTA latency including weight reload.** The simulator can model the AXI transfer of new patch-embed weights, but the realistic interaction with the host CPU running Python + a 3-gram LM beam search (≈60 GB RAM, per the DietCorp paper §5) needs a real host.

### 5.4 First-gen Edge TPU — what's available

You essentially skip the simulator question and run everything on the hardware. Useful instruments:
- The Edge TPU Compiler's static report (ops on/off TPU, on-chip cache usage, projected speedup).
- `tflite_runtime` interpreter + Coral's profiler hooks for per-op latency.
- USB current monitor for power.
- The first-gen Edge TPU's claimed throughput is 4 TOPS int8 at 2 W — DietCorp uses ~3.6 GMACs/s of that envelope on average, so you have orders of magnitude of headroom; reality will be bounded by data movement (USB 3 bandwidth between host and accelerator, ~250 MB/s sustained in practice), not compute.

---

## 6. Alternatives, in case Coral isn't the right venue

The brief asked me to be skeptical and to propose alternatives if Coral fails. The honest framing is: Coral isn't *failing*, it's just **one** datapoint along the deployment-portability axis the Slack post actually cares about. Here is a compact comparison.

| Platform | What it gives you that Coral doesn't | What it costs |
|---|---|---|
| **Coral USB Accelerator** ($60) | Cheapest real int8 wall-clock numbers; mature first-gen tooling | Limited ops; no on-device training; closed runtime |
| **Coral NPU sim (Verilator)** | Cycle-accurate inspection; open RISC-V toolchain | Matrix engine still in development; simulator-only — no partner silicon in your hands yet |
| **NVIDIA Jetson Orin Nano** ($249) | Real GPU + DLA on one board; full PyTorch (no TFLite needed); fp16 + int8; on-device backprop possible via standard PyTorch | Much higher power (5–15 W); fans; not what you'd put in a wearable, but a fair desktop proxy |
| **Hailo-8 / Hailo-8L** ($150–$300 modules) | Excellent Transformer support; published ViT/BERT numbers; HailoRT profiler is good | Closed toolchain; no public simulator; Linux host required |
| **Phone NPUs (Pixel Tensor TPU, Apple Neural Engine, Qualcomm Hexagon)** | The actual deployment target the Slack post named ("laptops/phones") | Each has its own toolchain (Core ML, QNN, NNAPI); harder to do controlled comparisons |
| **Laptop NPUs (Intel AI Boost, AMD XDNA, Apple Neural Engine)** | Same deployment target; large memory; can co-locate the 60 GB LM | Newer, less stable tooling; ONNX Runtime + DirectML or QNN is the portable path |
| **Renode** (Antmicro) | Whole-SoC co-simulation, RISC-V + peripherals | Not cycle-accurate; you'd model the NPU yourself |
| **gem5** | Research-grade CPU architectural detail | Doesn't model the NPU; major engineering lift to add it |
| **CPU-only TFLite + XNNPACK** | Trivially portable; gives you a baseline laptop/phone number | Slower than NPU runs; no NPU-specific insights |

**For the thesis I would suggest treating Coral as one of three platforms in a portability study**, not the centrepiece: (i) Coral NPU sim for cycle-level visibility, (ii) Coral USB Accelerator for real-silicon int8 wall-clock, (iii) Apple Neural Engine via Core ML on an iPhone for the actual phone-deployment number. Together they support the claim "DietCorp-Compact runs in real time on the kinds of int8 edge accelerators that consumer devices ship with."

---

## 7. The clinical and deployment context (why this matters beyond Coral)

The Slack post named the user-facing goal: restoring communication for ALS and brainstem-stroke patients. Three properties of that target shape the feasibility argument and should be in the thesis chassis around any Coral simulation:

1. **Latency is functional.** The DietCorp paper emphasises *streaming, real-time* decoding — i.e. text emitted as the patient is attempting to speak, not after the utterance ends. The reason the paper rejects bidirectional GRUs and post-hoc LLM merging is that they break the streaming property. Any "we made it run on edge X" claim has to preserve this. A 100 ms patch budget translates to: forward pass must finish in ≪ 100 ms on the deployment device, *every* patch, with no GC pauses. Edge NPUs are particularly good at this — no GPU memory thrashing, no Linux scheduler jitter — which is a real argument for the architecture choice that's worth making in the thesis intro.

2. **Distribution shift is the whole point of TTA.** The paper's Figure 2 shows WER deteriorating from 22.74 % to 32.58 % over 5 held-out days, and to 66.47 % over 8 days, *with no adaptation*; DietCORP cuts that to 21.24 → 22.97 % and 26.32 → 31.74 %. For an ALS patient this is the difference between "the BCI works for the first day after each calibration session" and "the BCI works continuously for weeks." This is why TTA can't be optional, and it's why insisting that the gradient step run on the patient's local device matters — sending neural data off-device for adaptation has privacy and reliability implications a clinical product must address.

3. **"Phones/laptops" is the deployment substrate, not Coral specifically.** The Coral feasibility analysis is a stand-in for the broader question "can we run DietCorp on a substrate the patient already owns?" Apple Neural Engine in particular is interesting because (a) iPhones are common assistive-tech substrates, (b) Core ML supports int8 and on-device personalisation, and (c) Apple's Neural Engine has been documented to run BERT-class Transformers in single-digit milliseconds. Investigating Coral *and* one phone NPU side-by-side is a stronger thesis story than Coral alone.

**A visual analogy for the streaming property:** think of DietCorp like a real-time-translation earpiece, not a recording-then-transcribe workflow. Every 100 ms of brain activity is the equivalent of a syllable, and the system has to emit its best phoneme guess before the next syllable arrives. Any architecture, any deployment substrate, any memory layer you add has to respect that drumbeat. ZenBrain's offline, sleep-consolidation, FSRS-scheduling machinery is fundamentally a *between-sessions* layer for that reason — it cannot live inside the syllable-to-syllable loop.

---

## 8. Minimum-viable experiment plan

A four-stage plan that gives a credible thesis chapter without overcommitting to one platform.

### Stage 1 — Reproduce DietCorp-Compact on the host (week 0–2)

- Clone https://github.com/ebrahimfeghhi/transformers_with_dietcorp (the paper's released code).
- Reproduce the 12.17 % WER with 3-gram LM and 18 ms TTA on your hardware.
- Add a numerical-stability layer: store all activations and weights in fp16 alongside the fp32 reference so quantization comparisons in stage 2 have a clean baseline.

### Stage 2 — Quantize and characterise on host CPU (week 2–4)

- Export to TFLite int8 with PTQ, using a representative dataset of ~500 validation trials.
- Measure WER degradation: target ≤ 1 % absolute. If worse, add QAT (one pass of fine-tuning with simulated quantization), targeting ≤ 0.5 % absolute.
- Profile on host CPU via `tflite_runtime`; record per-op CPU latency as the floor that NPU acceleration must beat.

### Stage 3 — Two-platform NPU comparison (week 4–8)

- **3a.** Coral NPU Verilator sim: compile the int8 model via LiteRT to the RISC-V `.elf`, run a single 10-second streaming inference, collect cycle counts per Transformer block, AXI traffic, and on-chip memory residency.
- **3b.** Coral USB Accelerator (first-gen): run the same TFLite int8 model with the edgetpu_compiler-targeted variant. Profile actual streaming latency over 1200 trials with `tflite_runtime`'s benchmark mode. Watch for op fallback in the compiler report; if LayerNorm or GELU falls back, document the resulting CPU-side time and propose a fused-op rewrite.

A positive Stage 3 result looks like: forward-pass-per-patch latency under 20 ms on Coral NPU sim (extrapolated to estimated silicon) and under 30 ms wall-clock on the USB Accelerator, with ≤ 1 % absolute WER degradation from quantization.

### Stage 4 — Memory-augmented variant + TTA loop (week 8–12)

- Add the tractable subset from §4.3: a static-shape N=8 KV cache + cross-attention gate sitting between the patch embed and the first Transformer block. Re-quantize, re-deploy.
- Build the DietCORP TTA host loop driving the NPU: 64 augmented forwards on-NPU, beam search on host, gradient step on patch-embed (+ gate) on host, weight reload to NPU. Measure per-trial total latency, target ≤ 50 ms.
- Run the 5-held-out-day and 8-held-out-day evaluation from the DietCorp paper. **Positive result:** the augmented variant matches or beats vanilla DietCORP's degradation curve (Figure 2 of the paper) without exceeding the streaming budget. **Negative result:** the augmented variant either underperforms on WER or breaks the 100 ms streaming budget — that's a real, publishable finding too, and tells you that the on-NPU memory tier added inference cost without enough TTA benefit.
- Bonus: replicate the comparison on one phone NPU (Apple Neural Engine via Core ML is the path of least resistance) to support the "phones/laptops" deployment claim.

### Metrics dashboard (what to record at every stage)

| Metric | Stage 1 | Stage 2 | Stage 3a (NPU sim) | Stage 3b (real EdgeTPU) | Stage 4 |
|---|---|---|---|---|---|
| WER 3-gram LM | ✓ | ✓ | — (functional check only) | ✓ | ✓ |
| WER 5-gram LM | ✓ | ✓ | — | ✓ | ✓ |
| Inference latency / patch | — | ✓ (CPU floor) | ✓ (cycles) | ✓ (wall-clock) | ✓ |
| TTA latency / trial | ✓ | ✓ | — | ✓ | ✓ |
| Peak memory (host) | ✓ | ✓ | — | ✓ | ✓ |
| On-chip memory residency | — | — | ✓ | partial (from compiler report) | ✓ |
| AXI / memory bandwidth | — | — | ✓ | — | ✓ |
| Power | — | — | — | ✓ (USB current meter) | ✓ |
| Held-out-day WER curve | — | — | — | ✓ | ✓ |

---

## 9. Things to verify or push back on

A few claims in the brief I'd flag to make sure the framing holds up:

- **"~8 MB on-chip SRAM"** — this is the first-gen Edge TPU's on-chip cache. The Coral NPU's on-chip storage is sized by the partner SoC and is currently configurable (default ITCM 8 KB / DTCM 32 KB in the sim, but real silicon will be larger). For a 9.4 MB int8 DietCorp model the first-gen number means most weights stream from host RAM over USB/PCIe per inference — that's the dominant cost on first-gen, not compute.
- **"int8 quantized, static shapes, limited control flow, no dynamic memory allocation"** — accurate for first-gen Edge TPU; mostly accurate for Coral NPU bare-metal programs. The Coral NPU's RISC-V scalar front-end does have control flow, but the executor model is "run to completion, no OS, no interrupts," which is operationally similar to static.
- **"Edge TPU has a reputation for being limited"** — fair, but for *this specific workload* (small streaming Transformer, int8, modest ops) it's not catastrophically limited. The DietCorp paper went out of its way to design a compact, causal, streamable architecture; that's exactly the kind of model Edge TPUs are best at, more so than a typical full-vocab LLM.
- **First-gen Edge TPU lacking any cycle-accurate simulator** is a documented gap, not a stylistic complaint. If your thesis chapter wants cycle-level insight, only the new Coral NPU sim (or alternatives like Renode/gem5) will give that.

---

## 10. What I'd want to discuss next

1. **Which Coral are we anchoring to?** If you want real wall-clock numbers for the thesis defense, the first-gen USB Accelerator is the practical choice. If you want a cycle-level architectural argument, the new Coral NPU Verilator sim is the right tool. They answer different questions.
2. **Is on-device TTA non-negotiable for the clinical narrative?** If yes, the host-CPU placement of the gradient step is the recommended path. If you can defer adaptation to a nightly offline pass, the deployment story simplifies enormously.
3. **How much of the "ZenBrain-inspired memory" is meant to be visible *inside* DietCorp's inference graph vs *around* it at session boundaries?** I've assumed mostly around; if you want it more inside, the working-memory KV-cache subset in §4.3 is the path, but acknowledge that you're then drawing on the *idea* of ZenBrain (multi-timescale, decay-aware memory) rather than on its actual implementation.
4. **What's the planned thesis platform shortlist?** If "laptops/phones" is the headline, Apple Neural Engine + Coral USB Accelerator is a strong pair; if "edge NPUs broadly," add Hailo-8 and a Jetson; if "open silicon," anchor on Coral NPU + a Renode SoC model.

If you can answer 1, 2, and 4, I can tighten Section 8 into an actual gantt-able plan with concrete commits and code.

---

## Sources

- Feghhi et al., "Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding" (arXiv:2507.02800v2, 2 Nov 2025). Attached PDF; key sections §3.5, §3.6, §3.7, §4.3, Tables 2–4, Appendix D Table 6, Appendix E.
- Bering, "ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems" (arXiv:2604.23878v1, 26 Apr 2026). Attached PDF; key sections §3 (Architecture), §4 (Key Mechanisms), §5 (PMA), Appendix B.1–B.11.
- [Coral NPU — Simulators](https://developers.google.com/coral/guides/software/simulator) (Google for Developers, last updated 2026-01-23)
- [Coral NPU — Architecture overview](https://developers.google.com/coral/guides/architecture) (last updated 2025-11-05)
- [Coral NPU — ML inferencing engines](https://developers.google.com/coral/guides/hardware/arch_ml) (last updated 2026-01-21)
- [Coral NPU — Platform components](https://developers.google.com/coral/guides/components) (last updated 2025-10-23)
- [Coral NPU — Compilers overview](https://developers.google.com/coral/guides/software/compilers) (last updated 2025-11-18)
- [Coral NPU — LiteRT converter for TensorFlow models](https://developers.google.com/coral/guides/software/lite-rt-delegate) (last updated 2026-02-12)
- [Coral NPU — FAQ](https://developers.google.com/coral/guides/faq) (last updated 2026-02-12)
- [TensorFlow models on the Edge TPU (first-gen)](https://coral.ai/docs/edgetpu/models-intro/)
- [Edge TPU Compiler reference](https://www.coral.ai/docs/edgetpu/compiler)
- [Post-training integer quantization | Google AI Edge](https://ai.google.dev/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant)
