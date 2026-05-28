import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from .encoder import BIT_Transformer
from .projector import MLPProjector

class ModalityAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, neural_embeds, text_embeds):
        batch_size = neural_embeds.size(0)
        neural_embeds = F.normalize(neural_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        tau = torch.clamp(self.temperature, min=1e-4)
        logits = torch.matmul(neural_embeds, text_embeds.t()) / tau
        labels = torch.arange(batch_size, device=neural_embeds.device)
        
        # If batch_size is 1, cross_entropy is always 0. 
        # But we still return it for consistency in multi-GPU/larger batch settings.
        loss_n = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_n + loss_t) / 2

class BrainToTextE2E(nn.Module):
    def __init__(self, llm_name="Qwen/Qwen2.5-1.5B-Instruct", session_ids=None, quantize=True, patch_size=4):
        super().__init__()
        self.llm_name = llm_name
        
        # 1. Load LLM with quantization if requested
        attn_implementation = "sdpa"
        if quantize:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation=attn_implementation
            )
            self.llm = prepare_model_for_kbit_training(self.llm)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation=attn_implementation
            )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Neural Encoder and Projector
        self.neural_encoder = BIT_Transformer(session_ids=session_ids, patch_size=patch_size)
        
        if hasattr(self.llm.config, "text_config"):
            llm_dim = self.llm.config.text_config.hidden_size
        else:
            llm_dim = self.llm.config.hidden_size
            
        self.projector = MLPProjector(output_dim=llm_dim)
        
        # 3. LoRA Configuration (Zhang et al. 2025, Appendix table: r=8, alpha=32, dropout=0.2)
        # NOTE: removed "audio_projector" target — that module only exists in Aero.
        # Base Qwen2.5 has the standard 7 LoRA targets.
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.2,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        # CTC head for auxiliary loss during E2E. Loaded from CTC pretraining.
        self.ctc_head = nn.Linear(self.neural_encoder.embed_dim, 42)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.contrastive_loss_fn = ModalityAlignmentLoss()
        
        # Prompt Template
        self.prompt_start = "<|im_start|>user\n<neural_activity>\n"
        self.prompt_end = "\n</neural_activity>\ndecode the above neural activity into an English sentence:<|im_end|>\n<|im_start|>assistant\n"

    @property
    def dtype(self):
        return self.llm.dtype

    def forward(self, neural_data, labels=None, session_id=None, neural_lengths=None,
                phonemes=None, phoneme_lengths=None, return_contrastive=True,
                ctc_weight=0.3):
        batch_size = neural_data.size(0)
        device = neural_data.device

        # 1. Neural Encoding & Projection
        neural_tokens = self.neural_encoder(neural_data, session_id=session_id,
                                             neural_lengths=neural_lengths)
        projected_embeds = self.projector(neural_tokens) 
        projected_embeds = projected_embeds.to(self.llm.dtype)
        
        # 2. Prompt Embeddings
        start_inputs = self.tokenizer(self.prompt_start, return_tensors="pt", add_special_tokens=False).to(device)
        end_inputs = self.tokenizer(self.prompt_end, return_tensors="pt", add_special_tokens=False).to(device)
        
        start_embeds = self.llm.get_input_embeddings()(start_inputs.input_ids).repeat(batch_size, 1, 1)
        end_embeds = self.llm.get_input_embeddings()(end_inputs.input_ids).repeat(batch_size, 1, 1)
        
        combined_embeds = torch.cat([start_embeds, projected_embeds, end_embeds], dim=1)
        prefix_len = combined_embeds.size(1)
        
        loss = 0
        ce_loss = torch.tensor(0.0, device=device, dtype=self.llm.dtype)
        contrastive_loss = torch.tensor(0.0, device=device, dtype=self.llm.dtype)
        
        if labels is not None:
            # 3. THE FIX: Add EOS token and Tokenization
            labels_with_eos = [l + self.tokenizer.eos_token for l in labels]
            text_tokens = self.tokenizer(labels_with_eos, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
            label_embeds = self.llm.get_input_embeddings()(text_tokens.input_ids)
            
            full_embeds = torch.cat([combined_embeds, label_embeds], dim=1)
            
            # Create labels tensor initialized with -100 (ignores loss)
            full_labels = torch.full((batch_size, full_embeds.size(1)), -100, device=device)
            
            # Fill in the target IDs for the labels part
            target_ids = text_tokens.input_ids.clone()
            
            # A. Mask the Padding (so it stops predicting <pad>)
            # We use the attention mask to find actual padding tokens
            pad_mask = text_tokens.attention_mask == 0
            target_ids[pad_mask] = -100
            
            # B. Mask the Prompt (so it stops memorizing the instructions)
            # We initialize with -100, so everything is masked by default.
            # We then fill target_ids into full_labels starting at prefix_len.
            full_labels[:, prefix_len:] = target_ids
            
            # ADDITIONAL SAFETY: Mask the first token after the prefix if it's a space or common start token
            # to force the model to actually look at the neural data for the first real word.
            # full_labels[:, prefix_len] = -100 
            
            # 4. 2D key-padding attention mask. HF Qwen2 adds the causal triangle internally.
            seq_len = full_embeds.size(1)
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

            # Mask padded neural positions
            if neural_lengths is not None:
                patched_lengths = (neural_lengths + self.neural_encoder.patch_size - 1) // self.neural_encoder.patch_size
                start_len = start_embeds.size(1)
                projected_len = projected_embeds.size(1)
                for i in range(batch_size):
                    actual_neural_end = start_len + patched_lengths[i]
                    attention_mask[i, actual_neural_end : start_len + projected_len] = 0

            # Mask padded label positions
            attention_mask[:, prefix_len:] = text_tokens.attention_mask

            # 5. LLM forward — let HF compute CE with the canonical shift + ignore_index=-100
            outputs = self.llm(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                labels=full_labels,
            )
            ce_loss = outputs.loss

            # ─── ONE-SHOT DIAGNOSTIC (first training batch only) ───
            if not hasattr(self, "_debug_printed"):
                with torch.no_grad():
                    # Aero-1-Audio bug: outputs.logits = outputs[0] = LOSS when labels are passed.
                    # Re-run forward without labels to get real logits for inspection.
                    dbg_outputs = self.llm(
                        inputs_embeds=full_embeds,
                        attention_mask=attention_mask,
                    )
                    shift_logits = dbg_outputs.logits[..., :-1, :]
                    shift_labels = full_labels[..., 1:]
                    valid = shift_labels != -100
                    preds = shift_logits.argmax(dim=-1)
                    print(f"[DBG] prefix_len={prefix_len}, full_seq_len={full_embeds.size(1)}, ce_loss_batch={ce_loss.item():.4f}", flush=True)
                    print(f"[DBG] dbg_logits_shape={tuple(dbg_outputs.logits.shape)}", flush=True)
                    for b in range(min(2, batch_size)):
                        vm = valid[b]
                        n = vm.sum().item()
                        tgt_ids = shift_labels[b][vm][:20].cpu().tolist()
                        pred_ids = preds[b][vm][:20].cpu().tolist()
                        tgt_txt = self.tokenizer.decode(tgt_ids, skip_special_tokens=False)
                        pred_txt = self.tokenizer.decode(pred_ids, skip_special_tokens=False)
                        print(f"[DBG] s{b} label='{labels[b][:80]}'", flush=True)
                        print(f"[DBG] s{b} n_valid_tokens={n}", flush=True)
                        print(f"[DBG] s{b} target_ids[:20]={tgt_ids}", flush=True)
                        print(f"[DBG] s{b} target_decoded='{tgt_txt}'", flush=True)
                        print(f"[DBG] s{b} pred_decoded='{pred_txt}'", flush=True)
                    self._debug_printed = True
            # ───────────────────────────────────────────────────────

            # 6. CTC auxiliary loss (fp32 upcast for numerical stability)
            ctc_loss = torch.tensor(0.0, device=device, dtype=self.llm.dtype)
            if phonemes is not None and phoneme_lengths is not None:
                ctc_logits = self.ctc_head(neural_tokens.float())   # CTC needs fp32
                ctc_log_probs = nn.functional.log_softmax(ctc_logits, dim=-1).permute(1, 0, 2)  # (T, B, 42)
                patched_lengths = (neural_lengths + self.neural_encoder.patch_size - 1) // self.neural_encoder.patch_size
                ctc_loss = self.ctc_loss_fn(ctc_log_probs, phonemes,
                                             patched_lengths.cpu(), phoneme_lengths.cpu())
                ctc_loss = ctc_loss.to(self.llm.dtype)

            # 7. Optional Contrastive Loss
            if return_contrastive:
                if neural_lengths is not None:
                    patched_lengths = (neural_lengths + self.neural_encoder.patch_size - 1) // self.neural_encoder.patch_size
                    neural_pooled = []
                    for i in range(batch_size):
                        # Avoid empty slices
                        curr_len = max(1, patched_lengths[i].item())
                        neural_pooled.append(projected_embeds[i, :curr_len].mean(dim=0))
                    neural_pooled = torch.stack(neural_pooled)
                else:
                    neural_pooled = projected_embeds.mean(dim=1)
                
                # Pooled text embeddings (ignoring padding)
                text_mask = (text_tokens.attention_mask == 1).unsqueeze(-1)
                text_pooled = (label_embeds * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
                contrastive_loss = self.contrastive_loss_fn(neural_pooled, text_pooled)
                
            loss = ce_loss + ctc_weight * ctc_loss + contrastive_loss
            return loss, ce_loss, contrastive_loss, ctc_loss
            
        return combined_embeds

    def generate(self, neural_data, session_id=None, neural_lengths=None, max_new_tokens=100):
        self.eval()
        device = neural_data.device
        batch_size = neural_data.size(0)
        
        with torch.no_grad():
            neural_tokens = self.neural_encoder(neural_data, session_id=session_id)
            projected_embeds = self.projector(neural_tokens).to(self.llm.dtype)
            
            start_inputs = self.tokenizer(self.prompt_start, return_tensors="pt", add_special_tokens=False).to(device)
            end_inputs = self.tokenizer(self.prompt_end, return_tensors="pt", add_special_tokens=False).to(device)
            
            start_embeds = self.llm.get_input_embeddings()(start_inputs.input_ids).repeat(batch_size, 1, 1)
            end_embeds = self.llm.get_input_embeddings()(end_inputs.input_ids).repeat(batch_size, 1, 1)
            
            combined_embeds = torch.cat([start_embeds, projected_embeds, end_embeds], dim=1)
            
            attention_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)
            if neural_lengths is not None:
                patched_lengths = (neural_lengths + self.neural_encoder.patch_size - 1) // self.neural_encoder.patch_size
                start_len = start_embeds.size(1)
                for i in range(batch_size):
                    actual_neural_end = start_len + patched_lengths[i]
                    attention_mask[i, actual_neural_end : start_len + projected_embeds.size(1)] = 0

            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                max_new_tokens=25,          # paper §A.3: max_new_tokens=25
                do_sample=True,             # paper §A.3: nucleus sampling
                top_p=0.9,                  # paper §A.3: top_p=0.9
                temperature=0.7,            # paper §A.3: temperature=0.7
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            
            generated_texts = []
            for i in range(outputs.size(0)):
                text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                generated_texts.append(text)
                
        return generated_texts
