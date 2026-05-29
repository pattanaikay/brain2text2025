"""
stages/decoder/phi4mm.py
------------------------
Track C2: Phi-4-Multimodal-Instruct (5.6B) used as text-only decoder.

Speech+video pretrained by Microsoft. Note: uses fused QKV (qkv_proj),
not separate q/k/v projections. LoRA target list differs from Qwen!

⚠ Borderline for 6 GB VRAM. Use batch_size=1 + gradient_checkpointing for
  local testing (5 epochs only). Full training requires A100.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


LLM_NAME = "microsoft/Phi-4-multimodal-instruct"
LLM_DIM  = 3072


class Phi4MMDecoder(nn.Module):
    PROMPT_START = "<|user|>\n<neural_activity>\n"
    PROMPT_END   = "\n</neural_activity>\ndecode the above neural activity into an English sentence:<|end|>\n<|assistant|>\n"

    def __init__(
        self,
        llm_name:     str   = LLM_NAME,
        quantize:     bool  = True,
        lora_r:       int   = 8,
        lora_alpha:   int   = 32,
        lora_dropout: float = 0.2,
    ):
        super().__init__()
        if quantize:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name, quantization_config=bnb, device_map="auto",
                trust_remote_code=True,
            )
            llm = prepare_model_for_kbit_training(llm)
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        # Phi-4 uses fused qkv_proj — target list differs from Qwen
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
            lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(llm, lora_cfg)
        tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer   = tok
        self.hidden_size = LLM_DIM

    @property
    def dtype(self):
        return self.llm.dtype

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(self, projected_embeds, labels=None, neural_lengths=None,
                patch_size=4, ctc_loss_val=None, ctc_weight=0.3,
                contrastive_loss_val=None, contrastive_weight=1.0) -> dict:
        device = projected_embeds.device
        B      = projected_embeds.size(0)
        projected_embeds = projected_embeds.to(self.dtype)
        emb = self.get_input_embeddings()

        start_ids  = self.tokenizer(self.PROMPT_START, return_tensors="pt",
                                    add_special_tokens=False).to(device)
        end_ids    = self.tokenizer(self.PROMPT_END,   return_tensors="pt",
                                    add_special_tokens=False).to(device)
        combined   = torch.cat([
            emb(start_ids.input_ids).expand(B, -1, -1),
            projected_embeds,
            emb(end_ids.input_ids).expand(B, -1, -1),
        ], dim=1)
        prefix_len = combined.size(1)

        ce_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        if labels is not None:
            labels_eos  = [l + self.tokenizer.eos_token for l in labels]
            text_tokens = self.tokenizer(labels_eos, return_tensors="pt",
                                         padding=True, truncation=True,
                                         add_special_tokens=False).to(device)
            label_emb   = emb(text_tokens.input_ids)
            full_emb    = torch.cat([combined, label_emb], dim=1)
            full_labels = torch.full((B, full_emb.size(1)), -100, device=device)
            tgt = text_tokens.input_ids.clone()
            tgt[text_tokens.attention_mask == 0] = -100
            full_labels[:, prefix_len:] = tgt
            attn_mask = torch.ones((B, full_emb.size(1)), device=device, dtype=torch.long)
            attn_mask[:, prefix_len:] = text_tokens.attention_mask
            out     = self.llm(inputs_embeds=full_emb, attention_mask=attn_mask,
                               labels=full_labels)
            ce_loss = out.loss

        ctc_loss_val         = ctc_loss_val         or torch.tensor(0.0, device=device, dtype=self.dtype)
        contrastive_loss_val = contrastive_loss_val or torch.tensor(0.0, device=device, dtype=self.dtype)
        total = ce_loss + ctc_weight * ctc_loss_val + contrastive_weight * contrastive_loss_val
        return {"loss": total, "ce_loss": ce_loss,
                "ctc_loss": ctc_loss_val, "contrastive_loss": contrastive_loss_val}

    def generate(self, projected_embeds, neural_lengths=None, patch_size=4,
                 max_new_tokens=100) -> list[str]:
        self.eval()
        B, device = projected_embeds.size(0), projected_embeds.device
        with torch.no_grad():
            projected_embeds = projected_embeds.to(self.dtype)
            emb = self.get_input_embeddings()
            start_ids = self.tokenizer(self.PROMPT_START, return_tensors="pt",
                                       add_special_tokens=False).to(device)
            end_ids   = self.tokenizer(self.PROMPT_END,   return_tensors="pt",
                                       add_special_tokens=False).to(device)
            combined  = torch.cat([
                emb(start_ids.input_ids).expand(B, -1, -1),
                projected_embeds,
                emb(end_ids.input_ids).expand(B, -1, -1),
            ], dim=1)
            attn_mask = torch.ones(combined.shape[:2], device=device, dtype=torch.long)
            outputs   = self.llm.generate(
                inputs_embeds=combined, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, num_beams=5, do_sample=False,
                early_stopping=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
            )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def build(spec: dict, prev_shape: tuple) -> tuple:
    decoder = Phi4MMDecoder(
        llm_name     = spec.get("llm_name",     LLM_NAME),
        quantize     = spec.get("quantize",      True),
        lora_r       = spec.get("lora_r",        8),
        lora_alpha   = spec.get("lora_alpha",    32),
        lora_dropout = spec.get("lora_dropout",  0.2),
    )
    return decoder, (LLM_DIM,)
