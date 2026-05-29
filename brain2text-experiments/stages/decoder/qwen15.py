"""
stages/decoder/qwen15.py
------------------------
Decoder stage: Qwen2.5-1.5B-Instruct with QLoRA (the baseline LLM decoder).

This wraps BrainToTextE2E but lets the Stack inject a pre-built encoder
and projector instead of constructing them internally. The decoder stage
owns the full forward() and generate() so the training loop stays simple.

Usage in training loop:
    outputs = stack.decoder(
        neural_tokens    = enc_out,
        projected_embeds = proj_out,
        labels           = batch["text"],
        session_id       = batch["session_id"],
        neural_lengths   = batch["neural_lengths"],
        phonemes         = batch.get("phonemes"),
        phoneme_lengths  = batch.get("phoneme_lengths"),
    )
    # outputs = {"loss": ..., "ce_loss": ..., "ctc_loss": ..., "contrastive_loss": ...}
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LLM_DIM  = 1536


class Qwen15Decoder(nn.Module):
    """
    Standalone LLM decoder stage.  Takes projected neural embeddings as
    inputs_embeds, runs the LLM forward pass, and returns named losses.
    """

    PROMPT_START = "<|im_start|>user\n<neural_activity>\n"
    PROMPT_END   = "\n</neural_activity>\ndecode the above neural activity into an English sentence:<|im_end|>\n<|im_start|>assistant\n"

    def __init__(
        self,
        llm_name: str  = LLM_NAME,
        quantize: bool = True,
        lora_r:   int  = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.2,
    ):
        super().__init__()
        self.llm_name = llm_name

        if quantize:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name, quantization_config=bnb, device_map="auto",
                trust_remote_code=True, attn_implementation="sdpa",
            )
            llm = prepare_model_for_kbit_training(llm)
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16,
                device_map="auto", attn_implementation="sdpa",
            )

        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_dropout=lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(llm, lora_cfg)

        tok = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

        self.hidden_size = (
            llm.config.text_config.hidden_size
            if hasattr(llm.config, "text_config")
            else llm.config.hidden_size
        )

    @property
    def dtype(self):
        return self.llm.dtype

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def forward(
        self,
        projected_embeds,       # (B, T_proj, hidden_size)
        labels=None,            # list[str]
        neural_lengths=None,    # (B,) raw bin counts
        patch_size: int = 4,
        ctc_loss_val=None,       # pre-computed CTC loss tensor (from compose.py)
        ctc_weight: float = 0.3,
        contrastive_loss_val=None,
        contrastive_weight: float = 1.0,
    ) -> dict:
        device = projected_embeds.device
        B      = projected_embeds.size(0)
        projected_embeds = projected_embeds.to(self.dtype)

        start_ids = self.tokenizer(self.PROMPT_START, return_tensors="pt",
                                   add_special_tokens=False).to(device)
        end_ids   = self.tokenizer(self.PROMPT_END,   return_tensors="pt",
                                   add_special_tokens=False).to(device)

        start_emb = self.get_input_embeddings()(start_ids.input_ids).expand(B, -1, -1)
        end_emb   = self.get_input_embeddings()(end_ids.input_ids).expand(B, -1, -1)
        combined  = torch.cat([start_emb, projected_embeds, end_emb], dim=1)
        prefix_len = combined.size(1)

        ce_loss = torch.tensor(0.0, device=device, dtype=self.dtype)

        if labels is not None:
            labels_eos  = [l + self.tokenizer.eos_token for l in labels]
            text_tokens = self.tokenizer(labels_eos, return_tensors="pt",
                                         padding=True, truncation=True,
                                         add_special_tokens=False).to(device)
            label_emb   = self.get_input_embeddings()(text_tokens.input_ids)
            full_emb    = torch.cat([combined, label_emb], dim=1)

            full_labels = torch.full((B, full_emb.size(1)), -100, device=device)
            target_ids  = text_tokens.input_ids.clone()
            target_ids[text_tokens.attention_mask == 0] = -100
            full_labels[:, prefix_len:] = target_ids

            seq_len = full_emb.size(1)
            attn_mask = torch.ones((B, seq_len), device=device, dtype=torch.long)
            if neural_lengths is not None:
                plen = (neural_lengths + patch_size - 1) // patch_size
                s    = start_emb.size(1)
                p    = projected_embeds.size(1)
                for i in range(B):
                    attn_mask[i, s + plen[i] : s + p] = 0
            attn_mask[:, prefix_len:] = text_tokens.attention_mask

            out    = self.llm(inputs_embeds=full_emb, attention_mask=attn_mask,
                              labels=full_labels)
            ce_loss = out.loss

        ctc_loss_val         = ctc_loss_val         or torch.tensor(0.0, device=device, dtype=self.dtype)
        contrastive_loss_val = contrastive_loss_val or torch.tensor(0.0, device=device, dtype=self.dtype)
        total = ce_loss + ctc_weight * ctc_loss_val + contrastive_weight * contrastive_loss_val

        return {
            "loss":             total,
            "ce_loss":          ce_loss,
            "ctc_loss":         ctc_loss_val,
            "contrastive_loss": contrastive_loss_val,
        }

    def generate(self, projected_embeds, neural_lengths=None, patch_size=4,
                 max_new_tokens=100) -> list[str]:
        self.eval()
        device = projected_embeds.device
        B      = projected_embeds.size(0)
        projected_embeds = projected_embeds.to(self.dtype)

        with torch.no_grad():
            start_ids = self.tokenizer(self.PROMPT_START, return_tensors="pt",
                                       add_special_tokens=False).to(device)
            end_ids   = self.tokenizer(self.PROMPT_END,   return_tensors="pt",
                                       add_special_tokens=False).to(device)
            start_emb = self.get_input_embeddings()(start_ids.input_ids).expand(B, -1, -1)
            end_emb   = self.get_input_embeddings()(end_ids.input_ids).expand(B, -1, -1)
            combined  = torch.cat([start_emb, projected_embeds, end_emb], dim=1)
            attn_mask = torch.ones(combined.shape[:2], device=device, dtype=torch.long)

            outputs = self.llm.generate(
                inputs_embeds=combined, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, num_beams=5, do_sample=False,
                early_stopping=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, use_cache=True,
            )
        return [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        llm_name     : str   = "Qwen/Qwen2.5-1.5B-Instruct"
        quantize     : bool  = True
        lora_r       : int   = 8
        lora_alpha   : int   = 32
        lora_dropout : float = 0.2
    """
    decoder = Qwen15Decoder(
        llm_name     = spec.get("llm_name",     LLM_NAME),
        quantize     = spec.get("quantize",      True),
        lora_r       = spec.get("lora_r",        8),
        lora_alpha   = spec.get("lora_alpha",    32),
        lora_dropout = spec.get("lora_dropout",  0.2),
    )
    return decoder, (decoder.hidden_size,)
