"""
stages/decoder/whisper_qwen.py
------------------------------
Track C3: BIT → Linear(384,1280) + LN → frozen Whisper-large-v3 → Qwen2.5-1.5B

Fits on 6 GB RTX 4050: Whisper-large-v3 (~2.9 GB) + Qwen2.5-1.5B-4bit (~1.2 GB) ≈ 4.1 GB.

Science: tests whether passing neural features through Whisper's learned speech
manifold (even when frozen) helps Qwen2.5 decode them.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from transformers import WhisperModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


WHISPER_NAME = "openai/whisper-large-v3"
LLM_NAME     = "Qwen/Qwen2.5-1.5B-Instruct"
WHISPER_DIM  = 1280
LLM_DIM      = 1536


class WhisperQwenDecoder(nn.Module):
    """
    BIT projected embeddings (384) → bridge → Whisper encoder (frozen, 1280)
                                   → bridge → Qwen2.5-1.5B (LoRA)
    """
    PROMPT_START = "<|im_start|>user\n<neural_activity>\n"
    PROMPT_END   = "\n</neural_activity>\ndecode the above neural activity into an English sentence:<|im_end|>\n<|im_start|>assistant\n"

    def __init__(
        self,
        neural_dim:   int  = 384,
        whisper_name: str  = WHISPER_NAME,
        llm_name:     str  = LLM_NAME,
        quantize:     bool = True,
        lora_r:       int  = 8,
        lora_alpha:   int  = 32,
        lora_dropout: float = 0.2,
    ):
        super().__init__()

        # ── Bridge 1: neural → Whisper ─────────────────────────────────
        self.neural_to_whisper = nn.Sequential(
            nn.Linear(neural_dim, WHISPER_DIM),
            nn.LayerNorm(WHISPER_DIM),
        )

        # ── Whisper encoder (frozen) ────────────────────────────────────
        whisper = WhisperModel.from_pretrained(whisper_name)
        self.whisper_encoder = whisper.encoder
        for p in self.whisper_encoder.parameters():
            p.requires_grad_(False)

        # ── Bridge 2: Whisper → LLM ────────────────────────────────────
        self.whisper_to_llm = nn.Linear(WHISPER_DIM, LLM_DIM)

        # ── Qwen2.5-1.5B with LoRA ──────────────────────────────────────
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
        self.tokenizer  = tok
        self.hidden_size = LLM_DIM

    @property
    def dtype(self):
        return self.llm.dtype

    def _project(self, neural_tokens):
        # neural_tokens: (B, T_patch, 384)
        whisper_in  = self.neural_to_whisper(neural_tokens)   # (B, T, 1280)
        whisper_out = self.whisper_encoder(whisper_in).last_hidden_state  # (B, T, 1280)
        llm_in      = self.whisper_to_llm(whisper_out)        # (B, T, 1536)
        return llm_in.to(self.dtype)

    def forward(self, projected_embeds, labels=None, neural_lengths=None,
                patch_size=4, ctc_loss_val=None, ctc_weight=0.3,
                contrastive_loss_val=None, contrastive_weight=1.0) -> dict:
        device = projected_embeds.device
        B      = projected_embeds.size(0)
        # projected_embeds here is encoder output (B, T_patch, 384), NOT yet through Whisper
        llm_emb = self._project(projected_embeds)

        start_ids = self.tokenizer(self.PROMPT_START, return_tensors="pt",
                                   add_special_tokens=False).to(device)
        end_ids   = self.tokenizer(self.PROMPT_END,   return_tensors="pt",
                                   add_special_tokens=False).to(device)
        emb_layer = self.llm.get_input_embeddings()
        start_emb = emb_layer(start_ids.input_ids).expand(B, -1, -1)
        end_emb   = emb_layer(end_ids.input_ids).expand(B, -1, -1)
        combined  = torch.cat([start_emb, llm_emb, end_emb], dim=1)
        prefix_len = combined.size(1)

        ce_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        if labels is not None:
            labels_eos  = [l + self.tokenizer.eos_token for l in labels]
            text_tokens = self.tokenizer(labels_eos, return_tensors="pt",
                                         padding=True, truncation=True,
                                         add_special_tokens=False).to(device)
            label_emb   = emb_layer(text_tokens.input_ids)
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
        device = projected_embeds.device
        B      = projected_embeds.size(0)
        with torch.no_grad():
            llm_emb   = self._project(projected_embeds)
            start_ids = self.tokenizer(self.PROMPT_START, return_tensors="pt",
                                       add_special_tokens=False).to(device)
            end_ids   = self.tokenizer(self.PROMPT_END,   return_tensors="pt",
                                       add_special_tokens=False).to(device)
            emb_layer = self.llm.get_input_embeddings()
            combined  = torch.cat([
                emb_layer(start_ids.input_ids).expand(B, -1, -1),
                llm_emb,
                emb_layer(end_ids.input_ids).expand(B, -1, -1),
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
    """
    spec keys:
        neural_dim   : int   = 384   (must match encoder output dim)
        whisper_name : str   = "openai/whisper-large-v3"
        llm_name     : str   = "Qwen/Qwen2.5-1.5B-Instruct"
        quantize     : bool  = True
        lora_r       : int   = 8
        lora_alpha   : int   = 32
        lora_dropout : float = 0.2
    """
    neural_dim = spec.get("neural_dim", prev_shape[-1] if prev_shape else 384)
    decoder = WhisperQwenDecoder(
        neural_dim   = neural_dim,
        whisper_name = spec.get("whisper_name", WHISPER_NAME),
        llm_name     = spec.get("llm_name",     LLM_NAME),
        quantize     = spec.get("quantize",      True),
        lora_r       = spec.get("lora_r",        8),
        lora_alpha   = spec.get("lora_alpha",    32),
        lora_dropout = spec.get("lora_dropout",  0.2),
    )
    return decoder, (LLM_DIM,)
