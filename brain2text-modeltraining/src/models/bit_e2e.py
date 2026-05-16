import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# --- MONKEY PATCHES FOR AERO-1-AUDIO COMPATIBILITY ---
import transformers.modeling_utils
import transformers.models.qwen2_audio.modeling_qwen2_audio as modeling_qwen

# 1. Mock missing FlashAttention class (since we use SDPA)
if not hasattr(modeling_qwen, 'Qwen2AudioFlashAttention2'):
    class Dummy(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    modeling_qwen.Qwen2AudioFlashAttention2 = Dummy

# 2. Fix the 'recompute_mapping' and 'missing_keys' bug in tie_weights for custom models
original_init_weights = transformers.modeling_utils.PreTrainedModel.init_weights
def patched_init_weights(self, *args, **kwargs):
    original_tie_weights = self.tie_weights
    def safe_tie_weights(*t_args, **t_kwargs):
        # Exhaustive fix: strip ALL keyword arguments to ensure compatibility with custom tie_weights
        return original_tie_weights() 
    self.tie_weights = safe_tie_weights
    return original_init_weights(self, *args, **kwargs)
transformers.modeling_utils.PreTrainedModel.init_weights = patched_init_weights
# -----------------------------------------------------

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
        loss_n = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return (loss_n + loss_t) / 2

class BrainToTextE2E(nn.Module):
    def __init__(self, llm_name="lmms-lab/Aero-1-Audio-1.5B", session_ids=None, quantize=True):
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
        self.neural_encoder = BIT_Transformer(session_ids=session_ids)
        
        if hasattr(self.llm.config, "text_config"):
            llm_dim = self.llm.config.text_config.hidden_size
        else:
            llm_dim = self.llm.config.hidden_size
            
        self.projector = MLPProjector(output_dim=llm_dim)
        
        # 3. LoRA Configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "audio_projector", "gate_proj", "up_proj", "down_proj"], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        self.contrastive_loss_fn = ModalityAlignmentLoss()
        
        # Prompt Template
        self.prompt_start = "<|im_start|>user\n<neural_activity>\n"
        self.prompt_end = "\n</neural_activity>\ndecode the above neural activity into an English sentence:<|im_end|>\n<|im_start|>assistant\n"

    @property
    def dtype(self):
        return self.llm.dtype

    def forward(self, neural_data, labels=None, session_id=None, neural_lengths=None, return_contrastive=True):
        batch_size = neural_data.size(0)
        device = neural_data.device
        
        # 1. Neural Encoding & Projection
        neural_tokens = self.neural_encoder(neural_data, session_id=session_id) 
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
            # 3. THE FIX: Text Tokenization and Label Masking
            text_tokens = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to(device)
            label_embeds = self.llm.get_input_embeddings()(text_tokens.input_ids)
            
            full_embeds = torch.cat([combined_embeds, label_embeds], dim=1)
            
            # Create labels tensor initialized with -100 (ignores loss)
            full_labels = torch.full((batch_size, full_embeds.size(1)), -100, device=device)
            
            # Fill in the target IDs for the labels part
            target_ids = text_tokens.input_ids.clone()
            
            # 3a. THE FIX: Replace all padding tokens with -100 in the targets
            pad_token_id = self.llm.config.pad_token_id if self.llm.config.pad_token_id is not None else self.tokenizer.pad_token_id
            target_ids[target_ids == pad_token_id] = -100
            
            full_labels[:, prefix_len:] = target_ids
            
            # 4. Attention Mask (including padding masking for both neural and labels)
            attention_mask = torch.ones(full_embeds.shape[:2], device=device, dtype=torch.long)
            
            # Mask neural padding
            if neural_lengths is not None:
                patched_lengths = (neural_lengths + self.neural_encoder.patch_size - 1) // self.neural_encoder.patch_size
                start_len = start_embeds.size(1)
                for i in range(batch_size):
                    actual_neural_end = start_len + patched_lengths[i]
                    attention_mask[i, actual_neural_end : start_len + projected_embeds.size(1)] = 0
            
            # Mask text label padding
            attention_mask[:, prefix_len:] = text_tokens.attention_mask
            
            # 5. LLM Forward Pass
            outputs = self.llm(inputs_embeds=full_embeds, attention_mask=attention_mask, labels=full_labels)
            ce_loss = outputs.loss
            
            # 6. Optional Contrastive Loss
            if return_contrastive:
                if neural_lengths is not None:
                    patched_lengths = (neural_lengths + self.neural_encoder.patch_size - 1) // self.neural_encoder.patch_size
                    neural_pooled = []
                    for i in range(batch_size):
                        neural_pooled.append(projected_embeds[i, :patched_lengths[i]].mean(dim=0))
                    neural_pooled = torch.stack(neural_pooled)
                else:
                    neural_pooled = projected_embeds.mean(dim=1)
                
                # Pooled text embeddings (ignoring padding)
                text_mask = (text_tokens.input_ids != pad_token_id).unsqueeze(-1)
                text_pooled = (label_embeds * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
                contrastive_loss = self.contrastive_loss_fn(neural_pooled, text_pooled)
                
            loss = ce_loss + contrastive_loss
            return loss, ce_loss, contrastive_loss
            
        return combined_embeds

    def generate(self, neural_data, session_id=None, neural_lengths=None, max_new_tokens=100, top_p=0.9, temperature=0.7):
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
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            generated_texts = []
            for i in range(outputs.size(0)):
                text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                generated_texts.append(text)
                
        return generated_texts
