# Approach #7: NeuroMoE Framework

This approach integrates concepts from two state-of-the-art EEG research papers: **EEGMoE** (Gao et al., 2026) and **BrainStack** (Zhao et al., 2026), into the BIT (Brain-to-Text Integration Transformer) architecture.

## Key Architectural Changes

### 1. Specific and Shared Mixture-of-Experts (SSMoE)
Inspired by **EEGMoE**, we replaced the standard Feed-Forward Network (FFN) in each Transformer block with an SSMoE block. This allows the model to decouple domain-specific features from domain-shared features.

*   **Specific Expert Group**: Uses **Top-K Routing** (K=2) to select the most relevant experts for each token. This captures fine-grained, domain-specific details (e.g., subject-specific spiking patterns).
*   **Shared Expert Group**: Uses **Soft Routing** where all experts contribute to every token. This captures universal EEG features that generalize across different recording sessions and tasks.

### 2. Functionally Guided Regional Experts
Inspired by **BrainStack**, we added a regional processing layer before the global Transformer.
*   The 512-channel neural input is partitioned into **8 functional regions** (64 channels each), mimicking the brain's modular organization (e.g., Prefrontal, Motor, Occipital).
*   Each region is processed by a **Regional Expert** (lightweight spatio-temporal CNN).
*   An **Adaptive Expert Routing Gate** computes context-dependent weights ($\alpha_i$) to fuse these regional summaries into a global "meta-representation."

### 3. Integrated NeuroMoE Encoder
The final encoder output is a fusion of:
1.  **Global Context**: From the Transformer blocks with SSMoE.
2.  **Regional Context**: From the BrainStack-inspired regional experts.

---

## Mathematical Foundations

### 1. Mixture-of-Experts Routing (EEGMoE)

For an input token $x$, the router weights $g_x$ are calculated as:
$$g_x = W_e \cdot x$$

The activation probability $p_i(x)$ for each expert $i$ is:
$$p_i(x) = \frac{\exp(g_{x,i})}{\sum_{j=1}^{|E|} \exp(g_{x,j})}$$

The output of the **Specific MoE** is the weighted sum of the Top-K experts:
$$\text{SpecMoE}(x) = \sum_{i \in \text{TopK}} p_i(x) e_i(x)$$

The output of the **Shared MoE** (using all $F$ experts) is:
$$\text{ShareMoE}(x) = \sum_{i \in F} p_i(x) f_i(x)$$

The final **SSMoE** output is:
$$\text{SSMoE}(x) = \text{SpecMoE}(x) + \text{ShareMoE}(x)$$

### 2. Load-Balancing Auxiliary Loss

To prevent "expert collapse" (where a few experts handle all data), we use an auxiliary loss $L_{aux}$:
$$L_{aux} = E \sum_{i=1}^E h_i D_i$$
Where:
*   $E$ is the number of specific experts.
*   $h_i$ is the fraction of tokens allocated to expert $e_i$.
*   $D_i$ is the mean router probability for expert $e_i$ across the batch.

### 3. Adaptive Regional Fusion (BrainStack)

Regional features $F_i$ are aggregated using learned coefficients $\alpha_i$:
$$F_{meta} = \sum_{i=1}^N \alpha_i \cdot F_i$$
$$\alpha_i = \frac{\exp(h(F_i))}{\sum_{j=1}^N \exp(h(F_j))}$$
Where $h(\cdot)$ is a learnable scoring function.

### 4. Total Optimization Objective

The model is trained end-to-end minimizing:
$$L_{total} = L_{CE} + L_{contrastive} + \lambda L_{aux}$$
We set $\lambda = 1 \times 10^{-4}$ to ensure $L_{aux}$ regularizes routing without dominating the primary decoding tasks.

---

## Implementation Details in Approach #7
- **File Structure**:
    - `src/models/moe.py`: Implementation of `SSMoEBlock`, `RegionalExpert`, and `BrainStackRouter`.
    - `src/models/encoder.py`: Integration of MoE components into the `BIT_Transformer`.
    - `src/models/baseline.py`: Updated `BITModel` to support multi-loss training.
    - `scripts/train_e2e.py`: Training pipeline updated for NeuroMoE.
- **Hyperparameters**:
    - Specific Experts: 6
    - Shared Experts: 2
    - Top-K: 2
    - Regions: 8 (64 channels per region)
