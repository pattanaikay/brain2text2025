# 🛠️ Training Diagnostics & Debugging Summary: BIT BCI Model

## 1. Phase 2 (End-to-End) Debugging: WER Stuck at 1.0
**The Scenario:** During Phase 2 training on an A100, the Loss is steadily decreasing, but the Word Error Rate (WER) and Character Error Rate (CER) remain completely flat at `1.0000` (100% error) after 20 epochs.

**The Diagnosis:** The model is structurally intact (gradients are flowing), but it is either generating completely empty strings or a single repeating garbage token. 

**Actionable Fixes:**
1. **The Missing Weights Check:** Ensure `train_e2e.py` is explicitly loading the pretrained weights from Phase 1.5 (`best_per_model.pth`). If initialized from scratch, the LLM receives pure noise.
2. **The Empty String Check:** Add a print statement in the validation loop (`print(f"Target: {target} | Pred: {predicted}")`). If the model is predicting `""`, your generation config (like `max_new_tokens` or early `<eos>` triggering) is broken.
3. **Modality Projector Initialization:** Check if the MLP Projector weights are too large, which can blow out the LLM's attention mechanism. Consider applying LayerNorm or initializing with a very small variance.
4. **LoRA Gradient Check:** Run `e2e_model.print_trainable_parameters()` to verify that `requires_grad=True` is actively applied to the LoRA adapters, allowing the LLM's text generation to learn.

---

## 2. Phase 1.5 (CTC Fine-Tuning): Expected Convergence Metrics
**The Scenario:** The `train_ctc.py` script triggered Early Stopping at Epoch 513 out of 800. We need to verify if the model successfully learned phonetic mapping before proceeding to Phase 2.

**The Target Metrics for Success:**
1. **Validation PER (Phoneme Error Rate):** Must fall between **0.15 and 0.30 (15% to 30%)**. Even with 25% errors, the LLM in Phase 2 has enough phonetic context to correct the spelling. *Do not proceed to Phase 2 if PER is > 60%.*
2. **CTC Loss:** Should drop from the high 20s to the low single digits (e.g., 0.5 to 2.5), plateauing alongside Validation Loss.
3. **"Peakiness" (Visual Check):** If you run a test batch, the raw `argmax` tensor should display massive blocks of the `<BLANK>` token (usually index 0) separated by sharp, single-frame phoneme predictions (e.g., `0, 0, 0, 'h', 0, 0, 'eh'`).

---

## 3. Phase 1.5 Debugging: Blank Token Collapse
**The Scenario:** The CTC training run halted at Epoch 510, but the logs show a permanently frozen Training Loss (`3.745`) and a Validation PER stuck exactly at `80.3333%`.

**The Diagnosis:** The network suffered from **Blank Token Collapse**. It realized that guessing the wrong phoneme triggered massive mathematical penalties, so it learned the "safest" local minimum: outputting the `<BLANK>` token for every single time step.

**Actionable Fixes:**
1. **Verify SSL Weight Transfer:** The neural spikes are too noisy to map to phonemes from scratch. Ensure `model.load_state_dict(torch.load('best_ssl_model.pth'), strict=False)` executes correctly before CTC training starts.
2. **Verify CTC Sequence Lengths:** `nn.CTCLoss` breaks if fed padded sequences. Ensure your dataloader passes the *actual, unpadded* sequence lengths for both the neural input and the target text to the loss function.
3. **Lower the Learning Rate:** The `5e-4` LR used in Phase 1 is often too aggressive for Phase 1.5. Drop the CTC learning rate to `1e-4` or `5e-5` to gently adapt the SSL weights without shocking the network into collapse.