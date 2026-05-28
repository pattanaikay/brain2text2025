# A Comprehensive Reading of Nine Papers in Computational Neuroscience and Deep Learning

*An expert reading produced for research, organised into nine deep analyses. Each section is written in the first person as if the paper's author were explaining the work — applying the Feynman technique throughout, anchoring claims in direct quotations, and using visual analogies and mathematical exposition wherever they aid understanding.*

---

## Table of Contents

1. **Hierarchical Reasoning Model (HRM)** — Wang et al., Sapient Intelligence
2. **POSSM: Generalizable Real-Time Neural Decoding with Hybrid State-Space Models** — Ryoo, Krishna, Mao et al., Mila/UdeM
3. **EEGMoE: Domain-Decoupled Mixture-of-Experts for Self-Supervised EEG** — Gao, Wang, Zhao
4. **BrainStack: Neuro-MoE with Functionally Guided Expert Routing for EEG Language Decoding** — Zhao et al., UTS
5. **Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding** — Feghhi, Kaasyap, Hadidi, Kao, UCLA
6. **TopoNets: High-Performing Vision and Language Models with Brain-Like Topography** — Deb, Deb, Murty, Georgia Tech
7. **ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems** — Bering, Zensation AI
8. **A Cross-Species Neural Foundation Model for End-to-End Speech Decoding (BIT)** — Zhang, He et al., Columbia/Stanford
9. **iPhoneme: Brain-to-Text Communication for ALS Using ConformerXL Decoding** — Cha, Chun, Park, Taejae University

---

# Paper 1 — Hierarchical Reasoning Model (HRM)

*Wang, Li, Sun, Chen, Liu, Wu, Lu, Song, Abbasi Yadkori — Sapient Intelligence, 2025*

## Context & Motivation: Why I Built HRM

I want to start with a frank confession about a paradox at the heart of modern deep learning. We named the field *deep* learning because deeper networks — more layers stacked on top of one another — were supposed to be the engine of capability. Yet today's most celebrated architectures, the Transformers powering ChatGPT and Claude, are paradoxically *shallow*. As I wrote in the paper:

> "Deep learning, as its name suggests, emerged from the idea of stacking more layers to achieve increased representation power and improved performance. However, despite the remarkable success of large language models, their core architecture is paradoxically shallow."

Why does this matter? Because complexity theory tells us that fixed-depth Transformers belong to a computational class called **TC⁰** — circuits of constant depth with threshold gates. This is a *very small* class. It cannot, in principle, solve problems requiring polynomial-time algorithms. So when you ask a Transformer to do Sudoku or maze-solving — problems that demand genuine search and backtracking — it does not so much "reason" as it pattern-matches against memorised solutions. The popular workaround, **Chain-of-Thought (CoT)** prompting, is the model talking to itself in language tokens to *simulate* depth, like a person solving a calculus problem by writing out every step on a notepad. CoT works, but it is a crutch:

> "CoT for reasoning is a crutch, not a satisfactory solution. It relies on brittle, human-defined decompositions where a single misstep or a misorder of the steps can derail the reasoning process entirely."

This is the gap I set out to close. I wanted a system that reasons in its *latent space* — its internal vector representations — the way a human seems to. When you do mental arithmetic, you don't generate English tokens internally; you manipulate something more abstract. The brain manages this lengthy latent reasoning effortlessly, and it does so with an architecture that is *recurrent* and *hierarchical* across multiple timescales. That's what I tried to import.

## The Core Idea: Two Clocks, Coupled

Imagine a kitchen with a head chef and a sous-chef. The head chef thinks slowly, sets the overall menu and strategy — "tonight we are making bouillabaisse." The sous-chef works fast, executing the small steps — chop the onion, sweat the leeks, deglaze the pan. The head chef does not interfere on every chop. Instead, the sous-chef runs until a sub-task reaches a stable state ("vegetables are softened"), and only then does the head chef glance up, adjust the plan if needed ("now add the fish"), and let the sous-chef begin a new fast phase of execution.

This is HRM in a nutshell. Two recurrent modules — call them **H** (slow, abstract, "head chef") and **L** (fast, detailed, "sous-chef") — interact on different clocks:

> "It features two coupled recurrent modules: a high-level (H) module for abstract, deliberate reasoning, and a low-level (L) module for fast, detailed computations."

Mathematically, the dynamics unfold over **N** high-level cycles, each of length **T** low-level steps. So the total computation runs for **N × T** timesteps in a single forward pass. Let me write the update rules precisely. With input embedding `x̃ = f_I(x)`, the two modules update as follows:

$$
z^i_L = f_L(z^{i-1}_L,\; z^{i-1}_H,\; \tilde{x};\; \theta_L)
$$

$$
z^i_H = \begin{cases} f_H(z^{i-1}_H,\; z^{i-1}_L;\; \theta_H) & \text{if } i \equiv 0 \pmod{T} \\ z^{i-1}_H & \text{otherwise} \end{cases}
$$

The L-module updates *every* timestep, conditioning on its previous state, the *frozen* current H-state, and the input. The H-module updates only *every T-th* step. After all N × T steps, the output network reads off a prediction: `ŷ = f_O(z^{NT}_H)`.

Here is a visual of the timing structure:

```
H state:  ●─────────────────────●─────────────────────●─────────────────────●
                                ↑                     ↑                     ↑
                          (updates here,         (updates here,        (final readout)
                           uses final z_L)       uses final z_L)
                                
L state:  ●─●─●─●─●─●─●─●─●─●─●─R─●─●─●─●─●─●─●─●─●─●─R─●─●─●─●─●─●─●─●─●─●─●
          fast inner loop (T steps)│  fast inner loop  │  fast inner loop
                                   │  starts fresh     │
                                  Reset                Reset
            ← cycle 1 →            ← cycle 2 →           ← cycle 3 →
```

The L-module repeatedly converges to a local equilibrium *within each cycle* (given the current H-context), and then the H-module ingests that converged state and pushes the L-module into a new "computational phase" with a fresh equilibrium to find.

## The Magic Trick: Hierarchical Convergence

This is the most beautiful idea in the paper, and it solves a chronic problem with recurrent networks. The problem is: RNNs converge **too fast**. The hidden state settles into a fixed point, updates become vanishingly small, and the rest of the network's "thinking time" is wasted on near-zero updates. Effective depth collapses.

> "Although convergence is crucial for recurrent networks, standard RNNs are fundamentally limited by their tendency to converge too early. As the hidden state settles toward a fixed point, update magnitudes shrink, effectively stalling subsequent computation and capping the network's effective depth."

HRM dodges this with what I call **hierarchical convergence**. The L-module converges fast (good — it reaches a stable answer for the current sub-problem), but the H-module's update then *resets* the problem the L-module is solving. So the L-module gets to converge *again*, this time toward a different equilibrium. From the outside, the forward residual (the magnitude of state change per step) looks like a sawtooth:

```
L-module residual over time:

│ ╱│   ╱│   ╱│   ╱│   ╱│
│╱ │  ╱ │  ╱ │  ╱ │  ╱ │
│  │ ╱  │ ╱  │ ╱  │ ╱  │
│  │╱   │╱   │╱   │╱   │
└──┴────┴────┴────┴────┴───→ time
   ↑    ↑    ↑    ↑    ↑
   H updates here, restarting L's convergence
```

The H-module's residual, in contrast, decreases slowly and steadily. Compare this to a standard RNN, where the residual just decays to zero rapidly and computation stalls.

> "This mechanism allows HRM both to maintain high computational activity (forward residual) over many steps (in contrast to a standard RNN, whose activity rapidly decays) and to enjoy stable convergence."

This is exactly the brain's trick. Slow theta waves (4–8 Hz) and fast gamma waves (30–100 Hz) ride on top of each other, with slow rhythms gating fast ones — what neuroscientists call **theta-gamma coupling**. The slow rhythm provides stable context; the fast rhythm does the local computation; together they achieve nested, multi-scale reasoning.

## The Gradient Problem and Its Beautiful Solution

If you wanted to train this naively, you would use **Backpropagation Through Time (BPTT)**, unrolling all N × T steps and storing every hidden state for the backward pass. This costs O(T) memory in T, which is brutal — and biologically implausible (your brain does not keep a buffer of every neural state from 30 seconds ago to compute gradients).

I proposed a workaround grounded in the math of **Deep Equilibrium Models (DEQ)**. The idea: if the recurrent network actually converges to a fixed point, you don't need to unroll. You can compute gradients *at the fixed point itself* using the Implicit Function Theorem.

Let me show this concretely. Suppose during cycle *k*, the L-module reaches a fixed point `z*_L` satisfying:

$$
z^*_L = f_L(z^*_L,\; z^{k-1}_H,\; \tilde{x};\; \theta_L)
$$

Then we can rewrite the H-update with a compact mapping F:

$$
z^k_H = F(z^{k-1}_H;\; \tilde{x},\; \theta), \quad \theta = (\theta_I, \theta_L)
$$

At the H-module's fixed point z*_H, the Implicit Function Theorem gives the **exact gradient**:

$$
\frac{\partial z^*_H}{\partial \theta} = \left(I - J_F\Big|_{z^*_H}\right)^{-1} \frac{\partial F}{\partial \theta}\Big|_{z^*_H} \quad \text{(Equation 1)}
$$

where J_F is the Jacobian of F. The matrix `(I − J_F)^{-1}` is expensive to compute exactly, but using the **Neumann series expansion**:

$$
(I - J_F)^{-1} = I + J_F + J_F^2 + J_F^3 + \cdots
$$

I approximated this with only the first term (`≈ I`) — the so-called **1-step gradient**. This dramatically simplifies the computation and gives an elegant chain rule:

$$
\frac{\partial z^*_H}{\partial \theta_H} \approx \frac{\partial f_H}{\partial \theta_H}, \quad \frac{\partial z^*_H}{\partial \theta_L} \approx \frac{\partial f_H}{\partial z^*_L} \cdot \frac{\partial f_L}{\partial \theta_L}
$$

The practical consequence is delightful: gradients flow only through the *most recent* update of each module — backward through "output head → final z_H → final z_L → input embedding". Memory cost is **O(1)** instead of O(T). And, critically, this aligns with how the brain plausibly handles credit assignment:

> "Given that each module only needs to back-propagate errors through its most recent local synaptic activity, this approach aligns well with the perspective that cortical credit assignment relies on short-range, temporally local mechanisms rather than on a global replay of activity patterns."

## Deep Supervision and Adaptive Computation Time

Two more pieces complete the system. The first is **deep supervision**: I run HRM as a sequence of "segments," each a full forward pass, with the hidden state carried over but *detached* from the gradient graph between segments. So gradients from segment m+1 do not flow back into segment m. This gives the H-module more frequent feedback and acts as a regulariser — and biologically, it echoes how neural oscillations gate when learning happens in the brain.

The second is **Adaptive Computation Time (ACT)**, my way of giving the model "thinking, fast and slow." A small Q-head reads the H-module state and decides whether to halt or keep computing more segments:

$$
\hat{Q}^m = \sigma(\theta_Q^\top z^{mNT}_H), \quad \hat{Q}^m = (\hat{Q}^m_{\text{halt}}, \hat{Q}^m_{\text{continue}})
$$

This is trained with Q-learning. The reward is binary: 1 if the prediction at halt-time is correct, 0 otherwise. The system learns to halt early on easy problems and keep thinking on hard ones — mirroring the brain's dynamic allocation of cognitive effort between System 1 (automatic) and System 2 (deliberate) reasoning.

The system is stable in Q-learning thanks to three structural choices: AdamW (which enforces L∞-bounded weights), RMSNorm post-norm layers, and weight decay. Together these satisfy recent theoretical convergence conditions for deep Q-learning without needing replay buffers or target networks.

## What HRM Achieves

Now for the headline results — and they are striking. With only **27 million parameters** and only **1,000 training examples**, no pretraining, no CoT supervision, HRM achieves:

- **40.3%** on ARC-AGI-1 (vs. o3-mini-high's 34.5%, Claude 3.7 8K's 21.2%)
- **~55%** on Sudoku-Extreme 9×9 (vs. 0% for all CoT baselines tried)
- **~74.5%** on Maze-Hard 30×30 (vs. 0% for all CoT baselines)

These are not incremental gains. They are categorical wins on tasks where giant CoT models score *zero*. The reason is that Sudoku and large mazes need genuine search and backtracking — algorithmic depth that fixed-depth Transformers simply lack, and that CoT cannot inject without specific algorithmic training data.

## The Most Striking Discovery: Emergent Dimensionality Hierarchy

I want to close on what for me is the most exciting result. I asked: does HRM, after training, *spontaneously* recapitulate a known property of the cortex?

In the brain, there is an observable **dimensionality hierarchy**: higher-order cortical regions (prefrontal, anterior cingulate) operate in higher-dimensional neural state spaces than primary sensory regions. Posani et al. (2025) measured this in mouse cortex using the **Participation Ratio**:

$$
\text{PR} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
$$

where λᵢ are the eigenvalues of the covariance matrix of neural population activity. Intuitively, PR is high when variance is spread across many dimensions (rich, flexible representation) and low when variance is concentrated in a few directions (compact, stereotyped representation).

When I measured PR for HRM's modules after training on Sudoku:

- **z_L** (low-level module): PR = **30.22**
- **z_H** (high-level module): PR = **89.95**

The H-module operates in a substantially higher-dimensional space — exactly the hierarchical relationship seen in mouse cortex (ρ = 0.79 in Posani's data). Moreover, when I varied the number of tasks the model has to handle from 10 to 100, the H-module's dimensionality *scales up* with task variety, while the L-module's stays stable. The high-level module is precisely the one that needs flexible, context-dependent representation, and it expands accordingly.

The control experiment seals the result. In an **untrained** HRM (random weights, same architecture), the dimensionality hierarchy disappears: z_L has PR ≈ 42, z_H has PR ≈ 41 — basically identical. This is *learned* structure, not an architectural artifact. The hierarchy emerges through training.

> "This confirms that the observed hierarchical organization of dimensionality is a learned property that emerges through training, not an artifact of the model's architecture."

## What This Means

HRM suggests we have been thinking about scaling wrong. Adding more parameters horizontally to a fixed-depth Transformer can only get us so far. The brain achieves vastly more sophisticated reasoning with vastly fewer "parameters" by exploiting *temporal recurrence at multiple timescales* — depth-in-time rather than depth-in-layers. HRM is a concrete demonstration that an architecture explicitly modelled on this principle can outperform giant CoT systems on the hardest reasoning benchmarks, with three orders of magnitude fewer parameters and three orders of magnitude less data.

The limitations are real and worth naming. HRM is currently a sequence-to-sequence architecture without language pretraining; integrating it with LLMs (so it can solve symbolic reasoning *and* talk fluently) is open work. The "1-step gradient" approximation is mathematically aggressive — it works empirically, but a tighter theoretical understanding of why is still wanted. And the high-level/low-level split is conceptual, not anatomically literal; the brain has dozens of hierarchical levels and many more than two intrinsic timescales. HRM is a proof of principle, not a finished theory.

But as a proof of principle, it is a powerful one. The take-home for me is this: depth need not mean stacked layers. It can mean *iterated dynamics*. And if you let two coupled dynamical systems work on different clocks, you can get the brain's trick — deep effective computation with shallow physical depth — for free.

---

# Paper 2 — POSSM: Generalisable, Real-Time Neural Decoding with Hybrid State-Space Models

*Ryoo, Krishna, Mao, Azabou, Dyer, Perich, Lajoie — Mila / Université de Montréal / Columbia / Penn, NeurIPS 2025*

## The Problem That Won't Go Away

If you want to build a brain-computer interface that actually works in the clinic — letting someone with ALS control a cursor or speak through a machine — you face an unforgiving triangle of constraints. I described it bluntly in the paper:

> "Building towards these applications will require neural decoders that meet three requirements: (1) robust and accurate predictions, (2) causal, low-latency inference that is viable in an online setting, and (3) flexible generalization to new subjects, tasks, and experimental settings."

Until POSSM, these three were essentially mutually exclusive. Picture them as a Venn diagram with empty intersection:

```
       ┌─────────────────────┐
       │     Accurate &      │
       │   Generalising      │  ← Transformers (POYO, NDT-2)
       │  (Transformers)     │     but too slow for real-time
       └─────────────────────┘
                 ✗
       ┌─────────────────────┐
       │  Fast & Lightweight │  ← RNNs (GRU)
       │      (RNNs)         │     but rigid inputs, no transfer
       └─────────────────────┘
```

Transformers ace generalisation through large-scale pretraining and flexible tokenisation, but their **quadratic** attention complexity over time makes them too sluggish for the < 10 ms decoding latency a closed-loop BCI demands. RNNs are blazingly fast and lightweight but cannot accept inputs with different neuron counts, sampling rates, or session structures without architectural surgery and retraining. POSSM — pronounced "possum," and standing for POYO-SSM — is my attempt to live at the empty centre of that Venn diagram.

## The Architecture in a Picture

The core idea is a **hybrid**: keep the Transformer's flexible front-end for *spatial* tokenisation of spikes, but replace its recurrent-history machinery with a **state-space model** (SSM) that runs in true streaming O(1)-per-step time. Here is the data-flow:

```
    Spikes in a 50 ms chunk:
    spike1 → (UnitEmb(neuron_i), tspike)  ──┐
    spike2 → (UnitEmb(neuron_j), tspike)  ──┤  variable-length 
    spike3 → (UnitEmb(neuron_i), tspike)  ──┤  spike sequence X_t
    ...                                      │
                                             ▼
                            ┌─────────────────────────┐
                            │  Cross-Attention        │  compresses N
                            │  with learnable         │  spikes to 1
                            │  query vector q         │  fixed-size
                            └─────────────────────────┘  latent z(t)
                                             │
                                             ▼
                            ┌─────────────────────────┐
                            │  State-Space Model      │  h(t) = f_SSM(z(t), h(t−1))
                            │  (S4D / GRU / Mamba)    │  O(1) update per chunk
                            └─────────────────────────┘
                                             │
                                             ▼
                            ┌─────────────────────────┐
                            │  Output Cross-Attention │  queries last k
                            │  with behaviour queries │  hidden states
                            └─────────────────────────┘
                                             │
                                             ▼
                                     v_x, v_y predictions
```

The left edge inherits **POYO's tokenisation scheme**: each spike is a pair (neuron-ID embedding, timestamp), where the neuron-ID is a learnable D-dimensional vector and the timestamp is encoded with rotary positional embedding (RoPE). The key thing is that the model does not need a fixed neuron count. A new session with 47 neurons and a different mix of cortical regions just means new ID embeddings — no architectural changes.

Mathematically, for a 50 ms chunk with N spikes, the cross-attention compresses them into a single latent vector z(t):

$$
z^{(t)} = \text{softmax}\!\left(\frac{qK_t^\top}{\sqrt{D}}\right) V_t, \quad K_t = X_t W_k,\; V_t = X_t W_v
$$

where q is a single learned **query** vector (the same one used every chunk), and X_t is the matrix of spike tokens in the chunk. The query "summarises" the chunk into a fixed-size code regardless of how many spikes happened. This is the PerceiverIO trick from Jaegle et al., adapted for spike data.

Now the recurrent backbone takes over:

$$
h^{(t)} = f_{\text{SSM}}(z^{(t)},\; h^{(t-1)})
$$

This SSM is doing the temporal integration that attention would have done in a vanilla Transformer — but in O(1) memory and time, with a hidden state that accumulates the whole history. POSSM is compatible with several backbones: **S4D** (diagonal structured state-space models, fast and parallelisable), **GRU** (the classic recurrent unit), and **Mamba** (selective SSMs with state-dependent dynamics). All three work.

## Why This Hybrid Is the Right Compromise

Here is the intuition I keep returning to. Local temporal structure within a single 50 ms chunk (which spikes co-occurred, which neurons fired together) is complex, irregular, high-bandwidth — exactly the kind of thing attention is good at. But integration over *seconds* of recent history is a slower, more linear process that a recurrent state-update handles more efficiently. POSSM splits the labour: attention does *local intra-chunk* work; the SSM does *global inter-chunk* work.

Compare the inference cost. A vanilla Transformer over T chunks costs O(T²) attention operations and grows linearly in memory. POSSM costs O(T) operations total and **constant** memory per step. That's the difference between a system that scales with the length of a sentence (Transformer) and one that scales like a thermometer (POSSM) — every new chunk is a cheap, constant-cost update.

## Transfer Across Sessions, Subjects, and Species

This is the part that makes me grin every time I look at the results table. Pretraining POSSM on a heterogeneous dataset of monkey reaching tasks (148 sessions, 670 million spikes, 26,032 neural units across M1, PMd, S1) yields a model — o-POSSM — that generalises in remarkable ways.

There are two finetuning strategies for a new session:

**Unit Identification (UI):** Freeze the entire model. Only train new unit embeddings (and a session embedding) for the new neurons. Less than 1% of parameters are updated. The pretrained recurrent dynamics — what the model has learned about motor cortex *in general* — are preserved entirely.

**Full Finetuning (FT):** Start with UI for some epochs, then unfreeze the whole model. More expensive, more accurate.

Both work. On a held-out monkey C centre-out session, o-POSSM-GRU with full FT achieves **R² = 0.96**, outperforming everything else including POYO-1. On a brand-new monkey not seen during pretraining (Monkey T's centre-out reach), o-POSSM-S4D achieves **R² = 0.91** — matching or exceeding what dedicated single-session models can do.

The *killer* result, though, is **cross-species transfer**. We took o-POSSM, pretrained entirely on monkey reaching data, and finetuned it on a human dataset of imagined handwriting (Willett et al., 2021). The participant has ALS; they imagine writing letters, and neural activity from their motor cortex is recorded. POSSM, with monkey pretraining, **outperforms** models trained from scratch on the human data:

> "Pretraining on monkey motor-cortical recordings improves decoding performance on the human handwriting task, highlighting the exciting potential for cross-species transfer."

This is biologically deep. It says the *latent dynamics* of motor cortex — the geometry of how populations of motor neurons encode intended movement — share enough structure across primate species that a model can learn the structure from one species and reuse it for the other. The neurons are different, the brains are different, the task is different (reaching vs. handwriting), but the *representational manifold* is similar enough to share.

## Speed: The Headline Engineering Result

On a workstation NVIDIA RTX8000 GPU, POSSM-GRU runs inference at roughly **2.44 ms per 50 ms chunk**. On a CPU (AMD EPYC 7502), o-POSSM runs at about **5.65 ms per chunk**. The clinically relevant threshold for real-time BCI is < 10 ms latency. POSSM clears it comfortably:

> "POSSM's inference time is well within the optimal real-time BCI decoding latency of ≤ 10 ms, making it a viable option for real-time BCI decoding applications."

By comparison, POYO-1 — equally accurate but reprocessing a full 1-second history every chunk — runs roughly **9× slower** on GPU. NDT-2 is heavier still. POSSM matches their accuracy at a fraction of the inference cost. That is precisely the empty centre of the Venn diagram I drew at the start.

## The Speech Decoding Demonstration

To push the long-context limits, we also evaluated on a human attempted-speech dataset (Willett et al., 2023). Speech sentences are long — minutes of neural activity at a time. Transformer-based decoders here run into a wall: the quadratic complexity of attention over thousands of time chunks is prohibitive, and even getting them to *train* requires gymnastics. POSSM, with its constant per-chunk cost, decodes attempted speech with strong accuracy where attention models become computationally infeasible.

## Where POSSM Sits in the Bigger Picture

The model is, in essence, an instantiation of an old idea — that the brain itself is a hybrid of fast local processing (cortical microcircuits, attention-like binding) and slower global integration (recurrent feedback loops, persistent state). Modern Transformers picked up only the local-processing half and tried to compensate for the missing global-integration half by feeding the model ever-longer contexts and ever-more attention. POSSM goes the other way: put the local processing where attention belongs (within a chunk), put the global integration where recurrence belongs (across chunks). The result is a model that is small (8M parameters), fast (sub-10ms inference), accurate (matches state of the art), and *transferable across species*. For a clinical BCI, that is the holy grail.

The honest limitations are that POSSM is evaluated offline despite being designed for online use — we have not yet run it in a live closed-loop study; that the cross-attention front-end only handles spikes, not LFP or other modalities (an obvious extension); and that we have not yet stress-tested on truly novel cortical regions (the pretraining is all motor-system data).

---

# Paper 3 — EEGMoE: Domain-Decoupled Mixture-of-Experts for Self-Supervised EEG Representation Learning

*Gao, Wang, Zhao — Chinese Academy of Sciences, IEEE TNNLS, 2026*

## The Headache of EEG Models

If you have ever tried to build a deep learning model for EEG, you know the dirty secret: every dataset is its own kingdom. A model that works beautifully on emotion recognition from the DEAP dataset will face-plant on motor imagery from BCI Competition IV-2a. A model tuned for one subject's brainwaves often fails on another's. The literature is full of papers with names like "EmotionNet-DEAP-v3" — exquisite specialists, useless generalists.

The conventional wisdom said: just pool everything. Train one big model on lots of EEG datasets and let it figure out what is shared. This is what large-scale pretraining did for vision (CLIP, MAE) and language (BERT, GPT). Why not EEG?

I have a striking figure in the paper that shows why. We trained a standard Transformer alternately on three EEG tasks — emotion recognition, motor imagery, and mental workload — and tracked the *direction* of each gradient update in a 2D projection of parameter space. The picture is sobering: 

> "During the gradient update process, the ER task drives the model parameters to optimize predominantly in the lower right direction, while the MI classification task guides the optimization toward the upper left direction. In contrast, the mental workload detection task steers the optimization toward the upper right direction."

Three tasks, three opposed gradient directions. The model is being pulled in three directions at once, and the natural compromise is mediocre on all three. This is the dreaded **gradient conflict** of multi-task learning, and it is especially severe in EEG because:

> "EEG tasks inherently differ in their neural correlates, dominant frequency bands, and temporal dynamics. Such discrepancies naturally induce gradient conflicts during joint training, which explains the inconsistency in optimization directions."

The motor imagery task lives in the **mu rhythm** (8–13 Hz) over sensorimotor cortex. Emotion recognition involves **alpha** asymmetry over frontal regions. Mental workload taps **theta** at the midline. These are *different brain mechanisms*. Forcing one set of weights to encode all three is asking for compromise.

## The Insight: Decouple, Don't Unify

Recent EEG foundation models — LaBraM, BIOT, MMM — chase **unification**: get every dataset into a common input format, train one model that processes them identically. I argue this is the wrong move. Unifying the *format* is necessary but ignoring the *content differences* discards exactly the domain-specific information that downstream tasks need. So my counter-proposal is: unify the input format, but *decouple* the representations through the architecture itself.

The vehicle for this decoupling is **Mixture-of-Experts (MoE)**. The idea: instead of one big monolithic network, have many small expert networks. For each input, a *router* decides which experts to use. Different inputs (from different domains, tasks, subjects) get routed to different specialists. Conflict avoided.

But standard MoE has its own failure mode. If a single token can be routed to *any* expert, the model can lose the *commonality* across domains — the universal structure of EEG that *is* shared across tasks. So I designed a two-track expert system:

- **Specific experts** with Top-K routing — only the K most relevant experts are activated per token, encouraging specialisation
- **Shared experts** with soft routing — all experts contribute to every token, learning universal features

Together: **SSMoE** (Specific and Shared MoE).

## The Mathematics of the Routing

For the **specific MoE**, with E candidate experts, the router computes routing scores from each input token x:

$$
g_x = W_e \cdot x, \quad p_i(x) = \frac{\exp(g_{x,i})}{\sum_{j=1}^{|E|} \exp(g_{x,j})}
$$

These probabilities pick the Top-K experts to activate:

$$
\text{SpecMoE}(x) = \sum_{i \in \text{TopK}} p_i(x)\, e_i(x)
$$

The intuition: each expert is a two-layer MLP with GELU activation, and the router decides which expertise pool best suits this particular token. For motor imagery tokens, the router might activate experts that have specialised in mu-rhythm patterns; for emotion tokens, those that have specialised in frontal alpha asymmetry.

For the **shared MoE**, with F shared experts, the routing is *soft* — every expert contributes to every token, weighted by its probability:

$$
\text{ShareMoE}(x) = \sum_{i \in F} p_i(x)\, f_i(x)
$$

These experts always learn from *all* domains, capturing what is universal about EEG.

The final SSMoE output is just the sum:

$$
\text{SSMoE}(x) = \text{SpecMoE}(x) + \text{ShareMoE}(x)
$$

Visually, you can think of it as two parallel parliaments deliberating:

```
                            Input token x
                                │
            ┌───────────────────┴───────────────────┐
            ▼                                       ▼
   Top-K Router (specific)             Soft Router (shared)
            │                                       │
   Activates only K best                  All F experts
   experts from E candidates               contribute
            │                                       │
   ┌────────┼────────┐                ┌──────┬──────┼──────┐
   ▼        ▼        ▼                ▼      ▼      ▼      ▼
  e_a      e_b      e_c              f_1    f_2    f_3    f_4
   │        │        │                │      │      │      │
   └────────┼────────┘                └──────┼──────┼──────┘
            ▼                                ▼
       SpecMoE(x)                       ShareMoE(x)
            │                                       │
            └──────────────────► + ◄────────────────┘
                                 │
                            SSMoE(x)
```

## The 4-D Input: Embedding Brain Topography

A small but important detail. EEG signals are messy because they are sampled at different rates with different electrode montages across datasets. To make these compatible, I preprocess every recording into a unified **4-D tensor** with dimensions:

$$
X \in \mathbb{R}^{(T \cdot S_r) \times B \times H \times W}
$$

- **T**: time window length (seconds)  
- **S_r**: sampling rate  
- **B**: number of frequency bands (5: δ, θ, α, β, γ)  
- **H × W**: 2D brain map dimensions (electrode positions projected onto a scalp grid)

Crucially, the **H × W** dimensions preserve **spatial topography** — channels that are physically close on the scalp end up close in the tensor. This is a small but powerful biological prior: in the brain, nearby cortical regions tend to do similar things, so a 2D layout that preserves this neighbourhood lets the model use convolutional priors meaningfully.

The five frequency bands carry meaningful neuroscience:
- **δ (1–4 Hz)**: deep sleep, anaesthesia
- **θ (4–8 Hz)**: memory, attention, midline cognitive load
- **α (8–14 Hz)**: relaxed wakefulness; mu rhythm in this band for motor
- **β (14–31 Hz)**: active concentration, motor planning
- **γ (31–45 Hz)**: high-level cognitive processing, feature binding

So a single input is a stack of 2D "brain maps" across these five physiological bands. The model sees something like "what is the alpha-band activity at frontal electrodes over the past two seconds?" in a form that respects scalp topology.

## Self-Supervised Pretraining: The Reconstruction Trick

For pretraining I use **masked signal reconstruction** — the EEG analogue of BERT's masked language modelling. We randomly mask 40% of the embedded patches and the model has to reconstruct them. The loss is L1:

$$
L_1 = \frac{1}{S} \sum_{i=1}^{S} \| Z_i - \tilde{Z}_i \|
$$

Plus an **auxiliary load-balancing loss** to prevent experts from becoming idle or overloaded:

$$
L_{\text{aux}} = E \sum_{i=1}^{E} h_i D_i
$$

where h_i is the fraction of tokens assigned to expert i and D_i is the average routing probability. This penalises imbalanced expert usage and keeps the parliament functional.

The total pretraining loss is:

$$
L_{\text{pretrain}} = L_1 + \alpha L_{\text{aux}}, \quad \alpha = 10^{-4}
$$

## What This Buys You

After pretraining on six EEG datasets and fine-tuning on three held-out ones (DEAP, BCIC4-2a, STEW), EEGMoE outperforms all prior state-of-the-art:

- DEAP (valence): **+2.96%** over EEGFuseNet
- DEAP (arousal): **+4.18%** over EEGFuseNet
- BCIC4-2a (motor imagery): **+6.01%** over DeepCNN
- STEW (mental workload): **+3.21%** over the STEW baseline

More importantly, the ablation shows *both* expert groups are essential:

- Remove the specific (Top-K) experts → drop across all three datasets
- Remove the shared (soft) experts → also drop across all three

This is the empirical signature of decoupling working. Specialisation alone is not enough (you lose the commonality); shared-only is not enough (you lose the specialisation). You need both, and the additive combination is what gives the model its breadth.

The expert-activation patterns also reveal something neuroscientifically satisfying: different datasets activate different specific experts at high frequency. Motor imagery data lights up a different subset of experts than emotion data. The model has *learned* to recognise domain boundaries from the data, without anyone telling it which task each sample came from. This is decoupling discovered by the architecture itself.

## What It Tells Us

The deeper lesson is methodological. We have spent five years assuming that the path to good EEG foundation models runs through *unification* — one big model, one big training run, one universal representation. EEGMoE says: unification is the *interface*, not the *objective*. What you want is shared front-end formatting plus decoupled internal representations. The brain itself is built this way — V1 and Broca's area share architectural primitives (six cortical layers, similar microcircuits) but compute radically different things. EEGMoE imports this insight into the model design.

The limitations are honest. The number of experts is hand-set rather than learned. The routing is data-dependent but not *task*-aware in the supervised sense — the model has no privileged access to task labels during pretraining (which is the point), but it would be interesting to see what task-conditioned routing would do. And the 4-D input format requires careful preprocessing for any new EEG dataset; the model is not yet a drop-in general-purpose brainwave reader.

---

# Paper 4 — BrainStack: Neuro-MoE with Functionally Guided Expert Routing for EEG-Based Language Decoding

*Zhao, Zhou, Jiang, Cao, Ma, Shen, Li, Wang, Lin — University of Technology Sydney, 2026*

## The Brain Is Not a Bag of Electrodes

When most EEG models look at a 128-channel recording, they see 128 parallel time series. The brain, of course, sees nothing of the sort. The cortex is anatomically partitioned into functionally specialised regions — prefrontal, frontal, central, temporal, parietal, occipital — and each plays a distinct role in any cognitive task. Treating those 128 channels as homogeneous tokens, the way many modern Transformers do, throws away one of neuroscience's most robustly established facts.

I built BrainStack to put that fact back into the model. As I wrote:

> "Most existing EEG models implicitly assume a homogeneous cortical structure, processing the entire electrode montage as a single unified source. This assumption overlooks the brain's well-established functional modularity, where distinct cortical regions exhibit specialized dynamics and contribute differently to cognitive processes."

The application I care about is **silent-speech decoding** — translating EEG recorded while a person imagines saying a word into the actual text of that word. This is exactly the kind of task where regional functional structure matters intensely: motor regions plan articulation, temporal regions handle phonological processing, occipital regions handle the visual cue, frontal regions handle executive control. A model that mashes them all together loses the structure that makes the task solvable.

## The Architecture: Seven Regional Experts Plus a Global Conductor

BrainStack decomposes an EEG trial X ∈ R^(C×T) into seven anatomically defined regional subsets plus one global view:

```
Raw EEG (128 channels × T timesteps)
        │
        ├─→ Prefrontal channels    ──→ CNet expert φ_1
        ├─→ Frontal channels       ──→ CNet expert φ_2
        ├─→ Central channels       ──→ CNet expert φ_3
        ├─→ Left-Temporal channels ──→ CNet expert φ_4
        ├─→ Right-Temporal channels──→ CNet expert φ_5
        ├─→ Parietal channels      ──→ CNet expert φ_6
        ├─→ Occipital channels     ──→ CNet expert φ_7
        │
        └─→ All channels (global)  ──→ CTNet (Transformer)
                                         │
                                         ▼
                          ┌──────────────────────────┐
                          │ Adaptive Routing Gate    │
                          │ assigns weights α_i      │
                          └──────────────────────────┘
                                         │
                                         ▼
                          F_meta = Σ α_i · F_i
                                         │
                                         ▼
                                 Word prediction
```

Each regional expert is a lightweight CNet — a compact CNN with temporal convolution, depthwise spatial convolution within the region, and separable convolution. These are cheap because they only see Cₙ ≪ C channels (one slice of the brain, not all of it). The **global expert** is CTNet — a heavier hybrid CNN + Transformer that sees *every* channel and learns long-range, cross-regional dependencies.

The output decoding objective is:

$$
\hat{y} = \arg\max_k \left[ g\left( \sum_{i=1}^{N} \omega_i^y \cdot \phi_i(X_i) \right) \right]_k
$$

where ω_i^y is the learned importance weight for region i, and g is the fusion function.

## The Adaptive Routing Gate

Here is the elegance of the design. Each expert produces a representation F_i. The routing gate scores each expert with a small learned scoring function h(·):

$$
\alpha_i = \frac{\exp(h(F_i))}{\sum_{j=1}^{N} \exp(h(F_j))}
$$

These weights sum to one and are used to combine the expert outputs:

$$
F_{\text{meta}} = \sum_{i=1}^{N} \alpha_i \cdot F_i
$$

The crucial point is that α_i is *context-dependent* — it depends on the input. For one trial, the model might lean heavily on the occipital expert (visual processing of the cue word); for another, on the temporal experts (phonological imagery). The router learns to dynamically reweight which region's expertise is needed.

## Hierarchical Cross-Regional Distillation

This is the most novel idea in the paper. The seven regional experts see *only their region* — they have no view of the rest of the brain. The global expert sees everything. So we can use the global expert to *teach* the regional experts about cross-regional context they cannot see directly.

This is a form of **knowledge distillation**. With temperature T, we minimise the KL divergence between the global expert's softened logits and each regional expert's softened logits:

$$
L_{\text{distill}} = \sum_{i=1}^{N} \text{KL}\!\left(\text{Softmax}\!\left(\frac{F_{\text{global}}}{T}\right) \,\Big\|\, \text{Softmax}\!\left(\frac{F_i}{T}\right)\right)
$$

The intuition: the regional expert keeps its anatomically grounded specialisation but is *biased* toward producing outputs consistent with the global picture. This emulates a real neurobiological phenomenon — **top-down modulation**, where higher-order regions (prefrontal cortex, default mode network) shape the activity of sensory and motor regions through descending feedback. The global expert here plays the role of "top-down."

## The Multi-Objective Loss

The full training loss combines four terms:

$$
L_{\text{total}} = \lambda \cdot L_{\text{fused}} + \alpha \cdot L_{\text{global}} + \beta \cdot L_{\text{local}} + \gamma \cdot L_{\text{distill}}
$$

with **dynamic scheduling**: λ ramps up during training, while α and β are modulated by progress. This is a curriculum: start with global supervision (give the model a stable whole-brain anchor), then progressively shift weight onto the fused multi-expert prediction and the distillation pressure as the regional experts mature.

## The SS-EEG Dataset: A New Benchmark for the Field

A major contribution beyond the architecture is the **SilentSpeech-EEG (SS-EEG)** dataset. Compare it to prior silent-speech EEG benchmarks:

| Dataset           | Subjects | Vocab | Trials | Hours | Channels |
|------------------|----------|-------|--------|-------|----------|
| KaraOne          | 12 (8 usable) | 11 | 1,056 | ~4.5 | 64 |
| Thinking Out Loud | 10 | 4 | 2,236 | ~3.5 | 128 |
| **SS-EEG (Ours)** | **12 (10 usable)** | **24** | **60,000** | **~120** | **128** |

SS-EEG is roughly **30× larger** in trials and ~30× larger in hours than the previous best. The protocol is four-phase: rest (5 s) → read (1 s, target word shown) → preparation cue (0.2 s) → silent speech (1.5 s, subject imagines saying the word).

## Results: What Modularity Buys You

On SS-EEG word classification, BrainStack achieves 41.87% average accuracy. The strongest baseline, TCNet, gets 29.50%. The improvement is **+12.37 points** — a categorical jump that is not explained by parameters (BrainStack has 1.06M; LaBraM has 5.8M and gets only 18.29%; STTransformer has 2.78M and gets 28.28%).

The ablation reveals the architectural ingredients matter:

- **BrainStack Homo**: replace heterogeneous experts (CNets + CTNet) with identical CTNet copies. Result: 32.78% — much worse despite *more* parameters
- **BrainStack RoI5**: coarser 5-region anatomical partition. Result: 37.19% — worse by ~5 points

Both ablations point to the same conclusion: the **architectural diversity** (different expert types for different regions) and the **fineness of the anatomical partition** are doing real work. This is not just MoE; it is *anatomically informed* MoE.

## Regional Contributions Reveal Brain-Plausible Patterns

When we examined the learned routing weights across subjects, the patterns make neurobiological sense:

> "The Occipital expert exhibits the strongest positive correlation with subject-level accuracy (r = 0.56), highlighting the importance of visual-cortical activity for silent-speech decoding."

Why does occipital matter? Because the task starts by showing the target word visually. The visual processing of the cue *anchors* the subsequent silent articulation, and subjects whose models successfully exploit occipital signal do better. Temporal experts also carry consistently high weights — consistent with the role of superior temporal cortex in phonological processing. Meanwhile, the global expert shows a *negative* correlation (r = −0.50) with accuracy, suggesting that subjects who over-rely on the global view (i.e., whose models don't lean on the regional specialists) do worse.

This is interpretability for free. The routing weights effectively give you a *learned saliency map* over functional brain regions for each task.

## What BrainStack Tells the Field

The bigger message overlaps with EEGMoE's: monolithic Transformers are the wrong default for EEG. The brain's modular functional organisation is not a side detail — it is the architectural prior that makes EEG decoding tractable, especially for low-SNR cognitive tasks like silent speech. Build models that *respect that prior*, and you get accuracy, generalisation, and interpretability simultaneously.

The honest limitations: SS-EEG is collected from a single lab and 12 subjects; cross-lab generalisation has yet to be shown. The seven-region partition is based on standard 10–20 anatomical landmarks, but functional boundaries are not perfectly aligned with these (Broca's area sits at the prefrontal/frontal/left-temporal junction, for instance). And the model still struggles dramatically on some subjects (S05: 10%) — pointing to deep individual differences in silent-speech neural signatures that may require subject-adaptive routing as a next step.

---

# Paper 5 — Time-Masked Transformers with Lightweight Test-Time Adaptation for Neural Speech Decoding

*Feghhi, Kaasyap, Hadidi, Kao — UCLA, 2025*

## A Pragmatic Paper About a Real-World Problem

This paper is the most engineering-focused of the bunch, and I love it for that reason. While other groups are racing for incremental WER improvements on the Brain-to-Text benchmark — often through ensembles of bidirectional GRUs plus GPT-3.5 post-processing — we asked a more grown-up question. *What if you actually had to ship this in a clinic?*

The clinical constraints are unforgiving:

> "Speech neuroprostheses... should ideally satisfy several key criteria beyond accuracy. First, they be able to operate in a real-time 'streaming' fashion... Second, they should have low computational requirements to enable on-device inference and adaptation... Finally, decoding algorithms should be easily integrated with test-time adaptation methods that mitigate performance degradation across time."

Top entries on the Brain-to-Text benchmark beat the baseline GRU on WER, but they use *bidirectional* GRUs (need future neural activity → can't stream), ensembles of 10 models (compute-prohibitive on a wearable device), and an LLM post-processor fine-tuned on Switchboard (likely overestimating performance since LLMs have probably seen Switchboard in training). The accuracy gains were real but the deployment story was a mess.

So we set ourselves a tighter goal: improve the actual neural-to-phoneme decoder (the part *before* the language model), in a way that is **causal**, **fast**, **memory-light**, and **adaptable at test time**. Three contributions.

## Contribution One: Time-Masking

The baseline GRU overfits early in training. Pretty much every team noticed this. Our diagnosis: the trial-level data has enough idiosyncratic structure (silences, breath patterns, subject-specific quirks) that the model memorises rather than generalises.

The fix is **time-masking**: during training, randomly mask contiguous chunks of the input neural activity. The model never sees a clean trial. Over 50% of each trial gets masked on average. This forces the model to rely on the *non-masked* portions and to make robust predictions from incomplete data — exactly the kind of generalisation we want.

Time-masking has been used in speech ASR (SpecAugment, Park et al. 2019), but using it this aggressively on intracranial recordings is new. It is also surprisingly principled: by removing temporal chunks, we shatter the spurious within-trial correlations that the GRU was memorising, leaving only the genuine signal.

## Contribution Two: Replace the GRU with a Compact Transformer

The baseline GRU uses 32-bin input windows with stride 4 — meaning consecutive inputs **overlap by 87.5%**. The model is reprocessing nearly the same data over and over. Our diagnosis: the GRU's lossy memory makes this redundancy necessary; it can't trust its own history.

Transformers have *perfect* memory over a fixed context window. They don't need overlapping inputs. So we segmented the neural data into **non-overlapping** 5-bin (100 ms) patches and fed them to a unidirectional (causal) Transformer with 5 layers.

Architecturally:
- Input patches: F·T_in dimensional (256 features × 5 time bins = 1280)
- Patch embedding: LayerNorm → Linear → LayerNorm
- 5 Transformer blocks, each with relative positional embeddings (T5-style), causal masking, and standard self-attention + FFN
- CTC loss on the output logits

The attention with relative position bias:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top + B_{\text{rel}}}{\sqrt{d}}\right) V
$$

where B_rel is a learned bias matrix indexed by relative distance between patches, and a causal mask ensures each patch only attends to itself and earlier ones (essential for streaming).

The results are striking. The time-masked Transformer:
- **83% fewer parameters** than the baseline GRU
- **52% less peak GPU memory**
- **20% lower WER** with a 3-gram LM (12.17% vs ~15%)
- **26% lower WER** with a 5-gram LM (8.18% vs ~11%)

It is also dramatically faster to *calibrate* — important because a clinical user might re-calibrate the model many times per day.

This result is itself noteworthy because earlier Brain-to-Text Benchmark submissions reported that Transformers *underperformed* the GRU. Two ingredients were missing in those attempts: aggressive time-masking, and the right architectural details (T5-style relative positional embeddings, careful causal masking, patch-based input). With those in place, the Transformer wins decisively.

## Contribution Three: DietCORP — Test-Time Adaptation in One Gradient Step

Neural signals drift across days. The model that decoded yesterday's speech well will be a few percentage points worse tomorrow. The standard fix is **test-time adaptation (TTA)**: as new data arrives, fine-tune the model on the fly.

The leading prior method for handwriting BCIs is **CORP** (Fan et al. 2024): for each new trial, use the n-gram language model to *correct* the neural decoder's output (the LM provides a noisy "pseudo-label"), then take *multiple* gradient steps to train the GRU toward this pseudo-label, while also maintaining a replay buffer of past data to prevent forgetting.

CORP is effective but heavy. Multiple gradient steps per trial; a growing buffer of past data; gradients computed over the entire trial. We wanted something that worked in a single gradient step with no stored history.

The trick: we already have time-masking as a training augmentation. Why not use it at test time too? Generate **K time-masked versions** of the current trial. Compute the n-gram-corrected pseudo-label from the unmasked decoded output. Then with one batched gradient step, train the model so all K masked versions also produce that same pseudo-label.

This is **DietCORP**. One gradient step. No stored data. Augmentations replace replay.

Mathematically, for trial X with pseudo-label ỹ from the n-gram LM, the DietCORP loss is:

$$
L_{\text{DietCORP}} = \frac{1}{K} \sum_{k=1}^{K} L_{\text{CTC}}(M_k(X),\, \tilde{y})
$$

where M_k is the k-th time-masking operation. The model is encouraged to predict ỹ from many partial views of X — a kind of consistency regularisation that prevents catastrophic adaptation to noisy pseudo-labels.

## What This Gets You

The combined system (time-masked Transformer + DietCORP) achieves:
- **>20% reduction in WER** over the baseline
- Effective mitigation of cross-day performance drops
- Real-time streaming capability (causal, non-overlapping inputs)
- Dramatically lower compute footprint

This is the kind of work that does not get into a flashy press release — there is no "AGI" claim, no "breakthrough" rhetoric. But it is precisely the kind of result that determines whether speech BCIs make it from research demos into the lives of people with ALS or brainstem stroke. The clinical bottleneck is not "one more accuracy point on a benchmark"; it is *deployability* — and that lives in compute, latency, memory, and graceful adaptation. This paper attacks all four.

The limitations are clean. The work is still single-subject (the one ALS participant in the Brain-to-Text dataset), and the time-mask rate is hand-tuned (50% on average) — there might be better schedules. The Transformer architecture is small (5 layers, 5 patches at 100 ms each); whether scaling it up would yield further improvements is not explored here.

---

# Paper 6 — TopoNets: High-Performing Vision and Language Models with Brain-Like Topography

*Deb, Deb, Murty — Georgia Tech, ICLR 2025*

## The Question No One Was Asking

Look at a region of cerebral cortex under a microscope and one fact leaps out: neurons that do similar things sit physically close together. This is **topographic organisation**, and it shows up at every scale. In primary visual cortex, neurons tuned to similar orientations form **pinwheel patterns** — tiny rosettes where every direction of edge is represented in a smooth angular gradient. At the macro scale, **face-selective regions** like the fusiform face area cluster spatially; nearby cortex prefers faces, slightly further cortex prefers bodies, further still prefers scenes. And in language cortex, recent work shows that neurons with similar **temporal integration windows** (how far back in a sentence they integrate context) cluster together.

Topography is not a quirk. It is *one of the brain's most robust organisational principles*. So when you train a neural network — vision or language — there is a striking absence at the heart of the model: there is no spatial relationship between neurons. Unit 7 in a layer is not "near" unit 8 in any meaningful sense. You could permute the unit indices and the model would behave identically. The brain literally cannot do this because its neurons are anchored to physical positions on a 2D cortical sheet.

> "Unlike the brain, most artificial neural network (ANN) models lack any kind of systematic organization of units."

Prior attempts to inject topography into ANNs hit a wall. Either they got topography but lost accuracy (the "topographic tax") or they got accuracy but only weak topography. The question we asked: can we have both, across both vision and language models, at the scale of modern Transformers and large ConvNets?

## The Cortical Sheet: Mapping Weights to a 2D Map

Step one: define what "topography" even *means* for a Transformer layer. A linear layer maps i inputs to o outputs. We reshape its o×i weight matrix into a **cortical sheet** C ∈ R^(h×w×d), where:
- The **area** h×w = o (each spatial location is one output neuron)
- The **depth** d = i (the weight vector for that neuron)
- We pick h and w as close to each other as possible (minimise perimeter → maximise neighbours per neuron)

Each location on the h×w grid represents one "neuron"; its depth-d vector is the neuron's incoming weights. For ConvNets the same idea applies with the output channels arranged on the grid.

```
Linear layer weights W ∈ R^(256 outputs × 768 inputs)
                          │
                          ▼
              Cortical sheet C ∈ R^(16 × 16 × 768)
              
            ┌─────────────────────────┐
            │ N1 N2 N3 N4 N5 ... N16  │   each box = one neuron's
            │ N17 N18 ...             │   weight vector
            │ ...                     │   (768 deep)
            │                         │
            │                         │
            └─────────────────────────┘
            16 × 16 = 256 neurons total
            
    Neighbours on the grid should compute similar things
```

## TopoLoss: The Synaptic-Pruning Inspired Trick

Now the question: what makes a cortical sheet "topographic"? It is *smooth*. Nearby neurons should have similar weight vectors. Distant neurons can differ freely.

Here is where we got our biological inspiration. The brain achieves topography in development partly through **synaptic pruning**: an early excess of synapses is systematically pruned based on activity-dependent rules, leaving only the most useful connections. Pruning preferentially removes *noisy*, *high-frequency* connections — the ones that contribute irregularity to the local representation map.

TopoLoss imports this intuition. We *blur* the cortical sheet — a low-pass filter that suppresses high-frequency noise while preserving the smooth, large-scale structure. Then we require the *original* sheet to be similar to its blurred version. If the original is already smooth (topographic), the blurred version is nearly identical → high cosine similarity → low loss. If the original is noisy and unstructured, the blurred version differs a lot → low cosine similarity → high loss.

Mathematically, with downsampling factor φ_h, φ_w (we use 3):

$$
\text{Blur}(X, \phi_h, \phi_w) = f_{\text{up}}\!\left( f_{\text{down}}\!\left( X,\; \frac{h}{\phi_h},\; \frac{w}{\phi_w} \right),\; h,\; w \right)
$$

The TopoLoss is then negative cosine similarity, averaged over the N depth slices:

$$
L_{\text{topo}} = -\frac{1}{N} \sum_{i=1}^{N} \frac{C_i \cdot C'_i}{\|C_i\| \|C'_i\|}
$$

where C'_i is the blurred version of slice i. The total training loss combines this with the standard task loss:

$$
L_{\text{total}} = L_{\text{training}} + \tau \cdot L_{\text{topo}}
$$

The scalar τ controls topography strength. Higher τ = smoother weights = more topographic = potentially worse accuracy. The Pareto trade-off lives here.

## What Makes TopoLoss Work Where Others Failed

The key insight is that we are *not explicitly setting* the local correlation structure to match the brain. Prior methods like TDANN tried to match the brain's measured correlation-vs-distance curve. That is heavy-handed: it pins the model to a specific shape and overconstrains learning.

TopoLoss instead asks for *smoothness* — high-frequency suppression — and lets the topographic structure *emerge* from the interplay of the smoothness pressure and the task loss. The network discovers which neurons should be neighbours by itself. This is the difference between (a) demanding a specific organisation, and (b) creating a pressure that lets useful organisation emerge.

The brain similarly does not "compute" its topography from a target spec. It emerges from local wiring constraints (long axons cost energy) and activity-dependent plasticity.

## The Results: Vision and Language

**Vision models**: ResNet-18, ResNet-50, ViT-b32 trained on ImageNet with TopoLoss. On the smoothness-vs-accuracy Pareto frontier, TopoNets dominate the previous best (LLCNN, TDANN) — TopoNet-ResNet18 (τ=50) drops only 25% from baseline vs LLCNN's 41% drop, at comparable topography.

**Language models**: GPT-Neo-125M and NanoGPT trained with TopoLoss on Wikipedia / FineWeb-Edu. Topography is applied to the `c_fc` layer (the first FFN layer in each Transformer block), based on prior work suggesting FFN modules act as key-value memory storing world knowledge.

> "We applied TopoLoss in the c_fc layer of GPT-Neo. This choice was based on prior work... that has suggested that the feed-forward modules in GPTs act as key-value memory modules storing world knowledge. The c_fc modules encode the persistent representations (in contrast to transient representations in the attention matrix) making it the theoretically grounded target for inducing topography."

TopoNet language models are comparable to baseline on BLiMP (grammaticality judgments) at meaningful levels of topography. Most other topographic language approaches do far worse.

## The Three Brain-Like Properties That Emerge

This is where the paper gets really interesting. TopoNets don't just *look* topographic — they exhibit three functional signatures known from neuroscience.

### Signature 1: Reduced Effective Dimensionality

Brains have surprisingly low-dimensional neural representations relative to their physical neuron count. Higher-level visual cortex predicts behaviour and image identity using only tens of effective dimensions, not hundreds of thousands. Effective dimensionality is computed via the participation ratio:

$$
\text{Effective Dim} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}
$$

We separated two competing hypotheses for why this matters in models:
- **Hypothesis 1**: Lower-performing models have lower dimensionality (it's a side-effect of poor learning)
- **Hypothesis 2**: Topography itself drives dimensionality reduction

The data is clear. In *both* vision and language: no significant correlation between model performance and dimensionality (P > 0.05), but a strong negative correlation between topography (smoothness) and dimensionality (R = −0.85, P = 0.0034 for vision; R = −0.96, P = 0.0005 for language).

This resolves a long-standing theoretical question. Topography *causes* the dimensionality reduction. Brain-like efficiency follows from brain-like spatial organisation.

### Signature 2: Localised Feature Processing

The most beautiful single result. We measured **selectivity** of individual neurons for object categories (faces, bodies, places, words) using the standard t-statistic:

$$
t = \frac{\mu_c - \mu_o}{\sqrt{\frac{\sigma_c^2}{N_c} + \frac{\sigma_o^2}{N_o}}}
$$

where c is the target category and o is "other categories." Then we plotted selectivity *spatially* on the cortical sheet. In baseline ResNets, face-selective and place-selective units are scattered randomly across the sheet. In TopoNet-ResNets, they cluster into spatial *patches* — directly analogous to the **fusiform face area** and **parahippocampal place area** in the human ventral stream.

> "TopoNets also replicate the key topographic signatures observed in the brain's visual and language cortices."

We see the same in language: large-scale topographic biases for animacy, real-world size, and curvature emerge on the cortical sheet. These are *the same biases* observed in human ventral temporal cortex.

### Signature 3: Increased Efficiency Under Pruning and Downsampling

Topographic models can be aggressively pruned (set the smallest weights to zero) or downsampled (literally subsample the cortical sheet) and still retain most of their performance. Baseline models collapse under the same pressure. The reason is biological: a topographic representation is *spatially coherent* — local downsampling preserves the gist because nearby neurons compute similar things, so losing one is partially compensated by its neighbours. A random representation has no such compensation; lose a unit, lose an idiosyncratic function entirely.

This is, incidentally, the same reason why the brain is robust to localised lesions of cortical tissue. Damage that takes out a small patch removes only a local subset of functions, not random ones scattered throughout.

## Why This Matters

For neuroscience, TopoNets gives us the first scalable tool to test theoretical claims about topography. Previously, theorists could write down hypotheses about what topography *does* for computation, but they could not test them on systems large enough to do real tasks. Now they can: train a TopoNet ResNet50 or GPT, measure its dimensionality, selectivity, efficiency, and compare it directly with brain data. The paper does exactly this — TopoNets achieve higher BrainScore values (predict brain responses) than baseline ANNs.

For AI, TopoNets give us models that are *naturally compressible* — you can prune them or downsample them with much smaller accuracy drops than baseline networks. This is potentially huge for edge deployment.

The honest limitations: τ has to be tuned per architecture (no universal default); the smoothness measure is one specific metric (smoothness from pairwise correlation-distance curves), and other metrics of topography exist; TopoLoss is applied to selected layers, not uniformly across the network, and choosing those layers is somewhat ad hoc; and we have not yet shown that TopoNet language models are better at downstream tasks (just comparable on BLiMP).

---

# Paper 7 — ZenBrain: A Neuroscience-Inspired 7-Layer Memory Architecture for Autonomous AI Systems

*Bering — Zensation AI, 2026*

## The Wrong Metaphors

I want to start with a heretical thesis. The entire enterprise of "agent memory" in modern LLM systems is built on the wrong metaphors. MemGPT borrows from operating systems (virtual memory paging). Mem0 borrows from databases (CRUD operations with LLM-managed deletions). A-Mem borrows from note-taking (Zettelkasten cards with backlinks). All of these are *information storage* metaphors — about how to keep bits accessible.

But memory in the brain is not primarily about storage. It is about a complex dynamical process of **consolidation**, **decay**, **reconsolidation**, **selection**, and **forgetting** that has been studied empirically for 140 years. Hermann Ebbinghaus measured forgetting curves in 1885. Bartlett showed reconstruction effects in 1932. Schacter, Tulving, McGaugh, Squire, Nadel — half a dozen Nobel-quality bodies of work — have left us with detailed quantitative laws of how memories form, strengthen, weaken, and re-form. None of this is in agent memory systems.

> "Existing AI agent memory systems rely on system-engineering metaphors—virtual-memory paging, flat LLM-driven storage, or Zettelkasten-style notes—and none integrate empirically validated principles of consolidation, forgetting, and reconsolidation."

ZenBrain is my attempt to *invert* the design space. Instead of starting from CS metaphors and decorating them with biological flavour, start from the cognitive-neuroscience literature and build out a system that implements its laws.

## The Architecture: Seven Layers, Fifteen Mechanisms

ZenBrain has **seven memory layers**, each grounded in well-established cognitive neuroscience:

```
                       ┌────────────────────────────────┐
                       │     MemoryCoordinator          │
                       │                                │
   ┌───────────────────┴──────────┬─────────────────────┘
   │           │      │           │             │
   ▼           ▼      ▼           ▼             ▼
Working    Short-   Episodic   Semantic    Procedural   Core   Cross-Ctx
Memory     Term     Memory     Memory      Memory       Memory Memory
(~7 items) (session) (events)  (KG facts)  (skills)    (pinned) (sharing)
```

- **Working memory**: ~7 items, Miller's magic number, highest-priority access
- **Short-term**: session-bounded buffer that consolidates to longer-term layers at session end
- **Episodic**: timestamped concrete experiences ("what happened, when, where")
- **Semantic**: abstract knowledge as a graph of facts with consolidation strengths
- **Procedural**: learned skills and successful workflow patterns
- **Core**: persistent identity facts that never decay
- **Cross-context**: privacy-aware bridges between isolated domains

This layering comes directly from the multi-store model (Atkinson & Shiffrin, 1968), Tulving's episodic/semantic distinction (1972), and Cohen & Squire's procedural memory work (1980).

Each layer is orchestrated by a **MemoryCoordinator** that handles five operations: store, recall, consolidate, decay, review.

## The Fifteen Mechanisms: Nine Foundational + Six PMA

This is where ZenBrain's depth shows. Beyond the layers, fifteen specific algorithms implement specific cognitive principles. Let me walk through the most important ones because they are individually interesting.

### Two-Factor Synaptic Model

In the brain, synapses do not just have a *weight* w. They have a **consolidation state** — newly formed synapses are labile and easily overwritten, mature synapses are protected by molecular mechanisms (CaMKII autophosphorylation, PKMζ persistence). Zenke et al. (2017, 2025) formalised this as a two-factor model: each synapse carries weight w_ij and consolidation variance σ²_ij. The **Fisher Information proxy** I_ij = 1/σ²_ij measures how "important" the synapse is.

ZenBrain implements this on knowledge-graph edges. Each fact-fact edge has not just a weight but a consolidation variance. Mature edges (low variance, high Fisher Information) resist decay and penalise weight changes — mathematically equivalent to Elastic Weight Consolidation (EWC) for catastrophic forgetting protection. Newly added facts are easily updated; deeply established facts are hard to overwrite. This is exactly what synapses do.

### vmPFC-Coupled FSRS

The Free Spaced Repetition Scheduler (FSRS) is the open-source descendant of Anki — it schedules when to "review" a flashcard based on past performance, optimising for long-term retention. ZenBrain extends FSRS with a **prediction-error signal** coupled to the ventromedial prefrontal cortex's role in encoding:

$$
\text{PE} = 1 - \cos(c_{\text{prev}}, c_{\text{now}})
$$

The PE is the cosine distance between previous and current context vectors — high PE means "context has shifted unexpectedly." This biases the FSRS scheduler via a sigmoid re-encoding factor: when context is shifting, schedule reviews sooner; when context is stable, schedule them later. This implements McGaugh's empirical finding that surprise enhances memory consolidation.

### Simulation-Selection Sleep Loop

This is one of my favourite mechanisms. Sleep, in the brain, is a memory operation. During slow-wave sleep, the hippocampus *replays* recent experiences, and these replays drive a two-stage process: (1) the CA3 region generates candidate episodes (including counterfactual variants), and (2) the CA1 region scores them with LTP/LTD-style updates based on temporal-difference error and reward signals.

ZenBrain implements this as a periodic "sleep" cycle. A **CA3 simulator** assembles candidate replay sequences from real episodes ∪ counterfactual permutations. A **CA1 selector** then runs reinforcement learning on these candidates, scoring each with:

$$
\text{TAG}(e) = \alpha |\delta_{\text{TD}}| + \beta R_e + \gamma N_e
$$

where δ_TD is the temporal-difference error, R_e is the reward signal, and N_e is the novelty score. High-TAG memories get LTP'd (strengthened); low-TAG memories get LTD'd (weakened or pruned). The result is **37% stability improvement** and **47.4% storage reduction**.

### TripleCopyMemory

This is another biologically grounded gem. Schapiro et al. (2024) showed that the brain stores memories not as single copies but with **divergent decay dynamics** — multiple traces decay on different timescales, with the longest-lived trace dominating after weeks.

ZenBrain stores each memory in three copies:
- **Fast copy**: exponential decay with τ = 4 hours (short-term)
- **Medium copy**: exponential decay with τ = 14 days (consolidation window)
- **Deep copy**: *logarithmic growth* with τ = 7 days (long-term, gets stronger over time)

The retrieved strength is the max:

$$
S(t) = \max(S_f, S_m, S_d)
$$

At 30 days, an Ebbinghaus-only model retains essentially zero strength; TripleCopyMemory retains **S(t) = 0.912** — over 90%. This is a dramatic difference, and it comes directly from the architectural choice to have multiple decay channels.

### NeuromodulatorEngine: Four Channels of Brain Chemistry

Real cognition is shaped by neuromodulators — chemicals released into the brain that don't carry specific information but modulate the *parameters* of computation. ZenBrain models four:

- **Dopamine (DA)** — reward prediction, motivation; modulates learning rate
- **Norepinephrine (NE)** — alertness, surprise; modulates attention bandwidth
- **Serotonin (5HT)** — patience, long-term value; opposed to DA per Daw et al. (2002)
- **Acetylcholine (ACh)** — encoding gain; modulates consolidation patience

Each runs with a tonic baseline and 5-minute half-life phasic bursts. DA/5HT have an opposition coupling. The outputs — learning-rate, exploration-bias, consolidation-patience, attention — feed downstream engines, parameterising the rest of the system in a state-dependent way. When the agent is "excited" (high DA + NE), it learns faster but is more distractible; when "calm" (high 5HT), it consolidates patiently.

### Other Mechanisms

I'll only briefly mention the rest because the paper details each: **ReconsolidationEngine** (retrieved memories enter a labile state and can be edited or rewritten under prediction-error gating — Nader et al. 2000); **PriorityMap** with four-dimensional priority (saliency, emotion, reward, goal) and an **amygdala fast-path** (|v| > 0.6 ⇒ P ≥ 0.5, modelling fear-conditioning's privileged access); **StabilityProtector** (NogoA/HDAC3 analog: only PEs > 0.5 + 0.3·L·ρ can overwrite established memories); **MetacognitiveMonitor** for bias detection (Fleming & Dolan, 2012); **Bayesian confidence propagation** with calibrated uncertainty.

## Evaluation: The Most Rigorous Memory Benchmarks Available

ZenBrain is evaluated on three benchmarks plus six mechanism-level studies:

- **LoCoMo**: 1,986 multi-session conversational QA pairs across five categories
- **MemoryAgentBench**: five competency dimensions
- **MemoryArena**: cross-session causal dependencies
- **LongMemEval-500**: per-question isolated haystacks across six categories

The headline results are striking. On LongMemEval-500, ZenBrain wins **12/12 head-to-head judge comparisons** against letta, mem0, and a-mem (J̄ = 0.545 vs 0.485, 0.394, 0.414). Under LongMemEval's binary judge, ZenBrain reaches **91.3% of long-context-oracle accuracy** at **1/106th the per-query token budget**.

On the full 15-algorithm ablation, the picture is fascinating: under *moderate* difficulty (decay = 0.15, 45 days), individual algorithms look redundant (≤0.1% degradation per removal — the cooperative network masks individual contributions). Under *challenging* conditions (decay = 0.20, 50 days), 7 of 15 algorithms become individually significant (ΔQ from −25.5% to −93.1%). Under *stress* (decay = 0.25, 60 days), 9 algorithms become critical. The same algorithm transitions from "redundant" to "critical" as you stress the system — exactly the pattern you would expect from a cooperative biological network where redundancy is the normal state and individual mechanisms reveal their function only under load.

## The NoDecay Ablation: Forgetting Is a Feature

The single result that stuck with me: a **NoDecay** ablation, where the full ZenBrain runs but Ebbinghaus decay is disabled. ΔP@5 = 0.002 (negligible, P = 0.043, Cohen's d = 0.015). Yet the full system *wins 12/12 head-to-head judge comparisons* on LongMemEval.

What this tells us: **forgetting is not a cost; it is the selection pressure**. By systematically reducing the strength of unreviewed memories, ZenBrain's downstream layers — retrieval, prioritisation, confidence — get cleaner, more focused inputs. Removing forgetting doesn't hurt raw retrieval; it just makes everything else slightly worse because the system loses its mechanism for *focusing on what matters*. As the paper's epigraph puts it (in beautifully apt German):

> "Wer viel speichert, findet viel. Wer klug vergisst, findet das Richtige."  
> *Who stores much finds much. Who forgets wisely finds the right thing.*

## What This Suggests

I think ZenBrain is doing something philosophically important. The AI field has spent five years arguing about whether bigger context windows or better retrieval is the solution to "long memory." Both miss the point. The brain solved this problem 100,000 years ago, and its solution is neither *bigger storage* nor *better lookup*. It is a *dynamics of forgetting and consolidation* that pre-filters what is worth remembering. The brain forgets *most* of what it encounters by design, because forgetting is how it identifies signal.

ZenBrain is, frankly, a kitchen-sink architecture — 15 algorithms is a lot, and the value of each individually is hard to assess (the ablation shows the redundancy is structural). But the integrative thesis is bold and timely: agent memory is not a storage problem, it is a cognitive-neuroscience problem, and we should be importing the empirically validated dynamics rather than reinventing them with OS metaphors.

The limitations the author acknowledges are real. Some traces are synthetic. The paper is sole-authored with AI assistance, and the breadth (15 algorithms, 4 benchmarks, 6 PMA components) is unusually large for one author. The current version is two-dimensional (7 layers × 15 mechanisms); a third dimension (temporal depth, generativity, affective encoding) is left for follow-up. And while ZenBrain dominates LongMemEval and MemoryArena, on raw LoCoMo flat-P@5 it trades 2–3 points to letta and mem0 (which use more permissive retrieval).

---

# Paper 8 — A Cross-Species Neural Foundation Model for End-to-End Speech Decoding (BIT)

*Zhang, He, Fan, Liu, Yu, Le, Li, Linderman, Duncker, Willett, Mesgarani, Paninski — Columbia / Stanford / Microsoft / UW, ICLR 2026*

## The Cascade Problem

When you watch a state-of-the-art speech BCI translate someone's attempted speech into text, you are actually watching a cascade of three semi-independent systems pretending to be one. First, an RNN maps neural activity to phoneme posteriors. Second, those posteriors are fed to a 3- or 5-gram language model that scores phoneme sequences. Third, a beam search assembles the best-scoring sentence. Each stage is trained separately on a different objective. The neural decoder optimises **phoneme error rate (PER)**; the language model optimises perplexity over text; the beam search greedily picks high-likelihood sequences. None of these objectives is the actual target you care about — **word error rate (WER)** of the final sentence.

The result is a known pathology. As I wrote in the paper:

> "Lower phoneme error rates (PER) from the RNN do not always translate to lower word error rates (WER) when decoding with the n-gram model."

The RNN can get better at phoneme prediction without the sentence-level WER improving — or worse, getting worse. The cascade is fragile because the stages aren't jointly optimised toward a common goal.

This is the gap BIT (BraIn-to-Text) closes: a *single differentiable* neural network that maps neural activity all the way to text, trained end-to-end. The framework has three parts and several novel ideas.

## The Architecture

```
Spike counts (T, C)
        │
        ▼
┌───────────────────────────────────┐
│ Patch embedding:                  │   group T_patch bins into a patch,
│ (T/T_p, C × T_p)                  │   project to transformer dim
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Transformer neural encoder        │   pretrained with self-supervised
│ (bidirectional attention, RoPE)   │   masked modelling on 367 hours
│ — pretrained cross-species —      │   of human + monkey neural data
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Shallow MLP projector             │   maps neural embeddings into the
│ (Linear → ReLU → Linear)          │   audio-LLM's embedding space
└───────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────┐
│ Audio-LLM decoder                 │   "decode the above neural
│ (with LoRA-adapted parameters)    │    activity into an English
│ + Modality Aligner (contrastive)  │    sentence:"  ← prompt
└───────────────────────────────────┘
        │
        ▼
   Decoded text
```

The neural encoder is a **Transformer** (not an RNN, departing from Willett et al.). The encoder takes neural data in shape (T, C) — T timesteps × C electrodes — groups every T_patch bins into a patch, projecting them to transformer tokens. Bidirectional attention over the patched sequence (with RoPE positional encoding) lets the model integrate context bidirectionally during training; for streaming inference one can use causal attention.

The cleverest design choice is **patch tokens** rather than per-bin tokens. Neural activity is recorded at 50 Hz; speech production happens at 30–60 words per minute ≈ 4–6 phonemes/second. The temporal mismatch between fine neural resolution and slower speech causes problems: per-bin tokens give a Transformer way too many positions to reason over for a short sentence. Patching aligns the model's effective timescale with the timescale of speech itself, while shortening the LLM context length.

## Self-Supervised Pretraining Across Species

The big technical bet is **cross-species pretraining**. We pretrained the Transformer encoder on **367 hours** of Utah array recordings — 98 hours human (across the Brain-to-Text '24 and '25 datasets plus additional speech and imagined speech) and **269 hours monkey** recordings during arm/motor tasks (Churchland, Chowdhury, Perich, Ma, et al.).

Each subject's data goes through a **subject-specific linear read-in / read-out layer** flanking the shared Transformer. This lets the model share most parameters across subjects while accommodating the different electrode counts and recording-quality variations.

The pretraining objective is **masked modelling** with MSE loss on reconstructed patches. We mask 15% of time patches and ask the model to fill them in.

The hypothesis: there is a *shared latent structure* in the way primate motor cortex encodes intended motor actions — whether the action is reaching, handwriting, or attempted speech — and a Transformer pretrained on lots of motor-cortical data of any kind will learn this structure and provide a useful prior for speech decoding.

The empirical answer is yes. In the cascaded setting (Transformer encoder → CTC phoneme posteriors → 5-gram LM → beam search), the pretrained encoder establishes new state-of-the-art on both the Brain-to-Text '24 and '25 benchmarks.

## The End-to-End Innovation: LLMs as Decoders

This is where the paper gets bold. After pretraining and fine-tuning the encoder for phoneme decoding, BIT removes the cascade entirely. The phoneme-aware Transformer encoder's outputs are projected via a shallow MLP into the embedding space of an **audio LLM** (a multimodal LLM pretrained on text + audio). A prompt is inserted: *"decode the above neural activity into an English sentence:"*. The LLM generates the sentence autoregressively.

```
Neural embeddings   →   MLP projector   →   LLM text embedding space
                                                       │
                                                       ▼
                                       [prompt: "decode the above..."]
                                                       │
                                                       ▼
                                          autoregressive text generation
```

We apply **LoRA** (low-rank adaptation) to the attention and feed-forward projections of the LLM, fine-tuning a tiny number of parameters while keeping the bulk frozen.

To encourage the projected neural embeddings to actually live in the LLM's text space, we train a **modality aligner** with contrastive learning:

$$
L_{\text{contrast}} = -\log \frac{\exp(\langle z_n, z_t \rangle / \tau)}{\sum_j \exp(\langle z_n, z_t^{(j)} \rangle / \tau)}
$$

where z_n is the mean-pooled neural embedding for an utterance and z_t is the mean-pooled text embedding for the matching transcript. The denominator includes contrastive negatives. This pulls matching neural-text pairs together while pushing non-matching ones apart in embedding space — essentially the CLIP objective adapted for brain-to-text alignment.

## The Audio LLM Surprise

Here is the most counter-intuitive result. We tested several decoders:
- A *text* LLM (Vicuna-7B variants)
- A small **audio-LLM** (Qwen-Audio-Chat, 8.5B parameters)
- The baseline RNN+LLM from Feng et al. 2024

The audio LLM, despite being smaller than some text-LLM alternatives we tried, *dramatically outperformed* the text-LLM decoder. Word error rate on the Brain-to-Text '24 benchmark dropped from **24.69%** (prior end-to-end SOTA) to **10.22%** with the audio LLM.

> "Notably, we find that small-scale audio-LLMs markedly improve end-to-end decoding."

Why audio? Because the encoder is pretrained to produce phoneme-aware embeddings — features that carry phonological information. An audio-LLM has been pretrained on actual speech-audio sequences and has internalised the *phonological* structure of language at a deeper level than a text-only LLM has. So the bridge from "neural activity → phonological embeddings → speech-aware decoder" is shorter and more semantically aligned than the bridge from "neural activity → phonological embeddings → text-only decoder."

This is a beautiful piece of cross-modal reasoning. Attempted speech in motor cortex is, at the level of representation, closer to *audio speech production* than it is to *written text*. The audio LLM is the right "ear" for the brain.

## Attempted vs Imagined Speech: A Cross-Task Win

The truly impressive result is what happens with **imagined speech** — internal speech without any attempted articulation. This is the long-term goal of speech BCIs because some patients cannot even attempt to speak. The imagined-speech dataset from Kunz et al. is tiny: 500 sentences for participant T12, 712 for T15.

When you fine-tune BIT on imagined speech, the pretrained encoder + audio-LLM gives strong performance, even though the task is data-poor. We also did an interpretability analysis showing that BIT's neural embeddings for attempted and imagined speech *align through shared semantic structure* — they map to nearby regions of the LLM's embedding space when they convey the same content. This **cross-task alignment** is the deep finding: attempted and imagined speech, while behaviorally different, share enough neural representational structure that a model trained primarily on the former generalises to the latter.

## What BIT Tells Us About Scaling

I think the deeper lesson of BIT mirrors a broader trend across this whole reading list. The path to general-purpose neural decoders runs through:

1. **Heterogeneous pretraining** across subjects, tasks, and species (POSSM, BIT)
2. **Subject-specific input/output adapters** wrapping a shared core (BIT, POSSM)
3. **Modern architectures** (Transformers, SSMs) replacing legacy RNNs (TimeMasked, BIT)
4. **End-to-end optimisation** rather than cascades (BIT vs cascade systems)
5. **Leveraging pretrained foundation models** (audio-LLMs here, vision foundation models in other settings)

The era of training a custom RNN per subject per task is coming to an end. The future is large pretrained encoders + lightweight subject adapters + foundation-model decoders. BIT is one of the cleanest demonstrations of that whole pattern in the neural decoding literature.

The limitations: BIT has only been evaluated on two participants (T12 and T15); the audio-LLM dependency means the system is heavy compared to causal streaming alternatives like POSSM; and end-to-end training with LoRA on a 7B+ LLM is still expensive enough that real-time deployment on edge hardware is non-trivial.

---

# Paper 9 — iPhoneme: Brain-to-Text Communication for ALS Using ConformerXL Decoding

*Cha, Chun, Park — Taejae University, 2026*

## The Numbers That Motivated This Work

I want to start with the most powerful single fact in this paper. Around 222,000–250,000 people worldwide have ALS, and **78–93% of them develop dysarthria**. That leaves roughly **173,000 to 232,500 people unable to communicate through natural speech**. The number of those people who have received a speech-capable BCI implant *globally, in all clinical trials* is approximately **22 to 31**.

> "This four-order-of-magnitude gap between the affected population (∼10^5) and the number of recipients (∼10^1) reflects both the high cost of current systems—first trials range from $30,000 to $100,000 per patient, with targeted costs of $50,000–$60,000 requiring ECoG 256-array electrodes and 4–8× NVIDIA A100 GPUs for training—and the inadequacy of low-cost alternatives such as eye-tracking keyboards, which enable communication at only 5–10 words per minute and suffer from the Midas touch problem."

Five orders of magnitude. The mismatch between people who need this and people who have it is so vast that I find it morally disorienting. iPhoneme is one team's response: a system designed not just for accuracy but for *clinical accessibility* — CPU inference, real-time latency, and a thoughtful interaction layer that addresses the practical barriers to deployment.

## The Architecture: ConformerXL

The decoder is a heavily modified **Conformer** architecture — the original Conformer (Gulati et al. 2020) was designed for automatic speech recognition (ASR) and won by combining the global-context modelling of attention with the local-feature extraction of depthwise convolution in a macaron-style sandwich: FFN → MHSA → Conv → FFN with half-step residuals. Attention captures long-range dependencies; convolution captures local fine-grained patterns. Together they cover the full structure of speech.

We adapted this for **intracranial EEG (iEEG)** signals, which differ fundamentally from audio:

- **Much higher channel dimensionality**: 512 features vs ~80 mel bands in audio
- **Lower temporal resolution**: 50 Hz vs 16 kHz in audio
- **Different noise characteristics**: electrode drift, neural non-stationarity, cross-channel artifacts that audio simply doesn't have

The result is **ConformerXL** (192.9M parameters), built in three layers:

```
512-channel iEEG (T timesteps × 512 features)
        │
        ▼
┌────────────────────────────────────────────┐
│ Temporal Prenet                            │
│  - Multi-scale dilated convolutions        │  for neural jitter correction
│  - Bidirectional GRU                       │  and short-term smoothing
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ 8× Temporal Subsampling                    │  reduces sequence length
│  - GELU activations                        │  while keeping CTC stable
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ 12 Encoder Blocks (ConformerXL core)       │
│  Each block:                               │
│    Pre-RMSNorm                             │
│    FFN ½                                   │
│    Multi-Head Self-Attention               │  global context
│    Convolution Module                      │  local features
│    FFN ½                                   │
│    Pre-RMSNorm                             │
└────────────────────────────────────────────┘
        │
        ▼
   CTC logits over phonemes
```

The **temporal prenet** is the key innovation. iEEG signals exhibit *neural jitter* — small temporal misalignments between trials due to neural variability. Multi-scale dilated convolutions (with dilation factors 1, 2, 4, 8) provide a hierarchy of receptive fields that can smooth jitter across different timescales simultaneously. The bidirectional GRU adds an additional smoothing pass. Together, these correct for the kind of trial-to-trial temporal variability that pure attention struggles with on small neural datasets.

The **8× temporal subsampling with GELU** addresses a CTC stability problem. CTC loss is unstable when the input sequence is too long relative to the output sequence — gradients become very small or very large. By aggressively subsampling (8×), we shorten the input to a length more proportional to the phoneme sequence, which makes CTC well-behaved.

The **Pre-RMSNorm** (rather than Post-LayerNorm) stabilises training of the 12 deep encoder blocks. This is a now-standard improvement from the Llama line of LLM architectures, applied here to the neural-decoding setting.

## The 3-Stage Decoding Pipeline

Stage 1: ConformerXL produces CTC posteriors over phonemes.

Stage 2: A **6-gram phoneme language model**, trained on 3.1 million sequences from the CMU Pronouncing Dictionary, LibriSpeech, and the T15 data itself, scores phoneme sequences. Crucially, the LM operates at the **phoneme level** rather than the word level. This is unusual — most ASR systems do word-level LMs — but for a BCI it makes sense because the bottleneck is *phoneme prediction*, and a phoneme LM can correct phoneme-level errors directly.

Stage 3: **WFST (Weighted Finite State Transducer) beam search** with beam width 128, Optuna-tuned over 150 trials. The WFST formalisation (Mohri et al.) represents the decoding problem as graph search over a weighted automaton. The 6-gram LM provides the edge weights; the CTC posteriors provide the emission probabilities; beam search finds the highest-likelihood path.

## The Headline Numbers

On the T15 dataset (45 sessions, 8,071 trials, 256-channel iEEG from four speech motor cortex regions: ventral 6v, primary motor area 4, area 55b, dorsal 6v):

- **Phoneme accuracy: 92.14%** (PER = 7.86%)
- **Word accuracy: 73.39%** (WER = 26.61%)
- Roughly **3% above prior state-of-the-art** (Card et al. 2024's 89% baseline)
- **180 ms latency on CPU** (no GPU required for inference)

The CPU inference is the crucial deployment fact. The training cost is 4–8 A100 GPUs (substantial), but once trained, the system runs at clinically usable latency on a regular CPU. That is the engineering breakthrough that opens the door to lower-cost deployment.

## The Other Half: The iPhoneme Interface

The second contribution is on the *interaction* side. Eye-tracking-based text input — the conventional accessibility tool for ALS — suffers from the **Midas touch problem**: because your eye is simultaneously a sensor (looking at the world) and a pointer (clicking on letters), every fixation is a potential unintended click. The standard fix is **dwell-time selection** — require the eye to rest on a target for 500–1000 ms before triggering a click. This reduces typing to 5–10 words per minute.

iPhoneme proposes a **chorded gaze + silent-speech paradigm**. The eye does the *pointing* (look at the letter you want). The brain does the *confirmation* (silently articulate a trigger phoneme — your iEEG detects it). Decoupling pointing from selection eliminates the Midas touch problem without imposing dwell-time delays.

The trigger phoneme selection is principled. We searched over ARPAbet phonemes using a composite metric combining:
1. **Detection accuracy** of that phoneme in iEEG
2. **Natural speech frequency** in LibriSpeech (365M phonemes)
3. **Safety scoring** — phonemes that would not be confused with common spontaneous internal speech

The result is a small set of high-accuracy, low-confusion phonemes that the user can use as discrete "click" gestures, sharing the iEEG channel with the main text decoder.

## Why iPhoneme Is Important

This is the rare paper that takes BCI seriously as a *system*, not just a model. Most speech BCI papers report PER and WER and call it a day. iPhoneme builds the entire chain: from filter design (4th-order Butterworth bandpass [0.3, 300] Hz with 80 dB/decade rolloff for power-line interference removal), through Common Average Reference spatial filtering, through ConformerXL, through WFST beam search, through the eye-tracking interaction protocol, all the way to CPU inference at 180 ms latency.

The four-band physiological grounding is worth noting:
- **Theta (4–8 Hz)**: memory encoding, attentional gating
- **Alpha (8–13 Hz)**: cortical idle states, sensorimotor mu rhythms
- **Beta (13–30 Hz)**: motor planning, sustained cortical activation
- **Gamma (30–100 Hz)**: local cortical processing, *phoneme-specific articulatory commands*

The fact that the model receives all four bands and uses them differentially through the temporal prenet means it can pick up the phoneme-specific gamma activity that does the actual articulatory encoding, while using the theta/alpha/beta signals as broader contextual modulators.

## What's Genuine Progress and What's Caveat

iPhoneme's improvement of ~3 absolute points on phoneme accuracy is significant but not earth-shattering. The deeper value is the architectural template — the temporal prenet for jitter, the 8× subsampling for CTC stability, the Pre-RMSNorm for deep-stack stability, the phoneme-level LM and WFST beam search — which is more transferable to the next dataset than any specific number.

The honest limitations: this is single-subject (T15); the dataset has natural distribution shift between 2024 and 2025 sessions that makes generalisation hard; and the 192.9M parameter model is much larger than POSSM or TimeMasked Transformers — the CPU-inference claim works for inference but training requires substantial GPU resources. The chorded gaze + silent-speech interface is also clever in principle but has not yet been tested in user studies — the paper does not yet show that the interface actually delivers usable typing speeds compared to dwell-based eye-tracking.

---

# Synthesis: What These Nine Papers Tell Us Together

Stepping back, these nine papers form a remarkably coherent picture of where computational neuroscience and deep learning are converging in late 2025 / early 2026. Let me trace the threads.

## Thread 1: The Brain Is the Architectural Prior

Five of the nine papers explicitly use neurobiological principles as architectural priors. **HRM** imports the brain's hierarchical, multi-timescale processing (theta/gamma coupling). **TopoNets** imports cortical topographic organisation. **BrainStack** imports functional regional modularity. **EEGMoE** imports the heterogeneity of EEG data sources. **ZenBrain** imports the entire cognitive neuroscience of memory dynamics. Each of these papers shows that respecting brain architecture *improves* model performance — there is no longer a topographic tax, a modularity tax, a hierarchical-depth tax. The brain's design choices are increasingly being shown to be near-optimal for the problems they evolved to solve.

## Thread 2: The Foundation Model Moment for Neural Data

Three papers (POSSM, BIT, EEGMoE) demonstrate that *cross-subject, cross-task, even cross-species* foundation models for neural data work. The recipe is now stable: a Transformer-style flexible front-end + lightweight subject/session adapters + a large heterogeneous pretraining corpus. POSSM and BIT both show **cross-species transfer** from monkey to human, which is biologically remarkable — it means primate motor cortex shares enough latent structure across species that pretraining on one transfers to the other.

## Thread 3: Mixture of Experts Is the Right Decomposition for Heterogeneity

EEGMoE and BrainStack both arrive at MoE architectures for the same reason: EEG data spans domains (tasks, subjects, datasets, brain regions) too heterogeneous for a monolithic model. EEGMoE decouples *task domains*; BrainStack decouples *anatomical regions*. Both show that an MoE built around the brain's functional decompositions outperforms larger monolithic alternatives. This is a strong methodological signal: as we scale to more diverse neural data, we will increasingly need architectures that *factor* the heterogeneity rather than averaging over it.

## Thread 4: Practical Speech BCIs Are Crystallising

Four papers (TimeMasked, BIT, iPhoneme, and POSSM in its speech-decoding extension) all attack the speech BCI problem with different strategies but converging insights:

- *Replace RNN with Transformer or hybrid* — RNNs were the legacy choice; modern architectures consistently win
- *Time-masking is a powerful regulariser* — heavy temporal masking forces robust phonological representations
- *Phoneme-level decoding + n-gram or LLM post-processing* — the consensus pipeline
- *Test-time adaptation matters more than chasing one more PER point* — neural signals drift across days, and graceful adaptation matters more than peak accuracy
- *End-to-end optimisation eventually wins* — BIT's end-to-end audio-LLM decoder is a glimpse of the cleaner future

These four papers together likely converge to a deployed clinical system within 2–3 years.

## Thread 5: Reasoning vs Memory vs Perception — Three Different Beasts

The reading list spans three different cognitive challenges, each requiring different architectural innovation:

- **Reasoning** (HRM): solved by *temporal recurrence at multiple scales*, not by stacking layers
- **Memory** (ZenBrain): solved by *cognitive-neuroscience dynamics*, not by bigger storage
- **Perception/Decoding** (POSSM, BIT, TimeMasked, iPhoneme, EEGMoE, BrainStack): solved by *flexible tokenisation + appropriate temporal models*

And **TopoNets** sits across all three, showing that brain-like organisation at the *unit* level is beneficial wherever you do representation learning.

## The Open Frontier

What is *not* yet solved that this list highlights?

1. **Unifying reasoning and language**. HRM solves reasoning but does not speak. LLMs speak but cannot reason in HRM's sense. Welding them is open work.

2. **Closed-loop deployment of speech BCIs**. The pieces are there (POSSM for streaming, BIT for end-to-end, TimeMasked for adaptation, iPhoneme for accessibility), but no paper here demonstrates an integrated clinically deployed system.

3. **Cross-modality foundation models for the brain**. The papers tackle EEG separately from intracranial recordings. A unified model that handles EEG, ECoG, Utah arrays, and even fMRI through a common abstraction is the obvious next step.

4. **Cognitive architecture for autonomous agents**. ZenBrain is a substantial step, but it focuses on memory. A full cognitive architecture combining HRM-style reasoning, ZenBrain-style memory, and BIT-style perception is the bigger goal.

5. **Theoretical understanding of why brain-like priors help**. We have abundant empirical evidence (TopoNets, HRM, BrainStack, EEGMoE) that brain-inspired architectures outperform brain-agnostic ones. The *theory* of why is still rudimentary.

I will finish where I started. The deepest lesson across these nine papers is that the brain is not just a *target* for neuroscience to understand — it is also an *architectural library* for AI to learn from. The systems we have spent fifty years studying with electrodes and fMRI machines turn out to encode design principles — hierarchical timescales, topographic organisation, modular specialisation, decay-and-consolidation dynamics, cross-region distillation — that translate directly into better-performing artificial systems. The two fields are no longer separate. They are co-evolving, and these nine papers are the fossil record of that co-evolution at a particular moment.

---

*End of comprehensive reading. Total length reflects the depth and complexity of the papers covered.*
