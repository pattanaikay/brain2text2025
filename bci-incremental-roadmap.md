# Incremental BCI Development Roadmap: Toward the BIT Framework

This document tracks the iterative transition from a baseline RNN-based model to the state-of-the-art **BIT (BraIn-to-Text)** architecture.

---

## 🟢 Approach 2: CNN + BiLSTM + N-gram (Current Baseline)
**Goal:** Improve temporal modeling and integrate linguistic constraints.

### 🛠️ Key Changes Needed
- **Architecture:** Swap `nn.GRU` for `nn.LSTM(bidirectional=True)`.
- **Data Engineering:**
    - Use both **Thresholded Spikes** and **Spike-Band Power (SBP)** features[cite: 731, 736].
    - Implement **Z-scoring across days** to handle session non-stationarity[cite: 152, 729].
- **Decoding:**
    - Replace Greedy Decoder with **Beam Search Decoder**[cite: 771, 772].
    - Integrate the character-level **3-gram Language Model** (trained on `transcription` keys).
- **Inference:** Implement `alpha` tuning to balance acoustic (neural) and language scores.

---

## 🟡 Approach 3: The Transformer Shift & SSL
**Goal:** Replace recurrence with attention and leverage unlabeled data.

### 🛠️ Key Changes Needed
- **Architecture:** - Implement a **Transformer Encoder** (7 layers, 6 heads, 384 embedding dim)[cite: 1059, 1092].
    - Implement **Time Patching**: Group 20ms bins into patches (e.g., patch size = 5) to align with speech timescales[cite: 159, 161].
- **Training Strategy (Two-Stage):**
    - **Stage 1 (Self-Supervised):** Train the encoder on **Masked Modeling** (MSE Loss) using all available T12 and T15 data (unlabeled)[cite: 182, 187].
    - **Stage 2 (Phoneme Fine-tuning):** Fine-tune the encoder for **Phoneme Decoding** using **CTC Loss**[cite: 191, 192].
- **Constraint:** Ensure subject-specific read-in/read-out layers are used during pretraining[cite: 128, 1059].

---

## 🟠 Approach 4: Modality Alignment
**Goal:** Teach the neural encoder to communicate with an LLM.

### 🛠️ Key Changes Needed
- **Architecture:** Add a **Shallow MLP Projector** (Linear -> ReLU -> Linear) to map neural embeddings to the LLM's text latent space[cite: 130, 167].
- **LLM Setup:** - Load **Gemma-2B** or **Aerol-Audio 1.5B** in 4-bit (bitsandbytes) to stay within 6GB VRAM[cite: 377, 546].
    - **Freeze the LLM** entirely during this phase.
- **Training Strategy:**
    - Use **Contrastive Learning (InfoNCE Loss)** to align neural modality tokens with ground-truth text tokens[cite: 131, 840, 841].
    - Implement **Mean Pooling** to create single "modality tokens" per trial[cite: 131, 837].

---

## 🔴 Approach 5: Final BIT Implementation (Full End-to-End)
**Goal:** Joint optimization of the encoder and decoder.

### 🛠️ Key Changes Needed
- **Architecture:** Unfreeze the LLM components using **LoRA (Low-Rank Adaptation)**[cite: 133, 176].
    - Apply LoRA to `q`, `k`, `v`, `o` projections and feed-forward layers[cite: 1123].
- **Inference Strategy:** - Switch from Beam Search to **Nucleus (Top-p) Sampling** for the LLM decoder to maintain inference speed[cite: 803, 807].
    - Use a prompt: `"decode the above neural activity into an English sentence:"`[cite: 132, 444].
- **Training Strategy:** - Multi-objective loss: **Cross-Entropy Loss** (for generation) + **Contrastive Loss** (for alignment)[cite: 202, 855].
- **Final Polish:** Implement an **LLM Merger/Ensemble** to synthesize results from models trained with different seeds[cite: 362, 809, 811].

---

## 📈 Metric Tracking
| Iteration | Model Type | Decoding Method | Target CER | Target WER |
| :--- | :--- | :--- | :--- | :--- |
| Approach 2 | CNN+BiLSTM | N-gram Beam Search | ~0.15 | ~0.25 |
| Approach 3 | Transformer | N-gram Beam Search | ~0.10 | ~0.20 |
| Approach 5 | BIT | Gemma-LoRA | <0.05 | ~0.10 |
