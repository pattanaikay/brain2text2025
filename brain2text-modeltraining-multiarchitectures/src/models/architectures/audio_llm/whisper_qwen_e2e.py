"""Arch-5b: Whisper-large-v3 encoder + Qwen2.5-1.5B decoder (split stack).

Neural embeddings are projected into Whisper's d_model (1280) space, fed through
the frozen Whisper encoder, then adapted to Qwen2.5 with a Linear(1280, 1536)
bridge.  Rationale: Whisper encoder has learned a speech manifold; projecting
neural features into it may improve the LLM prior (Arch-5 §5.2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    WhisperModel, WhisperConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from src.models.encoder import BIT_Transformer

WHISPER_NAME = "openai/whisper-large-v3"
QWEN_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class BrainToTextWhisperQwen(nn.Module):
    """Two-LLM pipeline: Whisper encoder manifold + Qwen2.5 decoder.

    Whisper encoder is frozen.  LoRA applied to Qwen2.5 only.
    """

    def __init__(self, session_ids=None, quantize=True, patch_size=4):
        super().__init__()
        # Whisper encoder (frozen) — only encoder half needed
        whisper_cfg = WhisperConfig.from_pretrained(WHISPER_NAME)
        whisper_full = WhisperModel.from_pretrained(WHISPER_NAME)
        self.whisper_encoder = whisper_full.encoder
        for p in self.whisper_encoder.parameters():
            p.requires_grad_(False)
        whisper_dim = whisper_cfg.d_model  # 1280

        # Neural encoder → Whisper manifold projection
        self.neural_encoder = BIT_Transformer(session_ids=session_ids, patch_size=patch_size)
        self.neural_to_whisper = nn.Sequential(
            nn.Linear(384, whisper_dim),
            nn.LayerNorm(whisper_dim),
        )

        # Qwen2.5-1.5B decoder
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ) if quantize else None
        self.llm = AutoModelForCausalLM.from_pretrained(
            QWEN_NAME,
            quantization_config=bnb_cfg,
            torch_dtype=None if quantize else torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if quantize:
            self.llm = prepare_model_for_kbit_training(self.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size  # 1536
        # Whisper → Qwen2.5 bridge
        self.whisper_to_llm = nn.Linear(whisper_dim, llm_dim)

        lora_cfg = LoraConfig(
            r=8, lora_alpha=32, target_modules=LORA_TARGETS,
            lora_dropout=0.2, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        self.ctc_head = nn.Linear(384, 42)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.prompt_start = "<|im_start|>user\n<neural_activity>\n"
        self.prompt_end = "\n</neural_activity>\ndecode the above neural activity into an English sentence:<|im_end|>\n<|im_start|>assistant\n"

    def _encode_neural(self, neural_data, session_id=None, neural_lengths=None):
        neural_tokens = self.neural_encoder(neural_data, session_id=session_id,
                                             neural_lengths=neural_lengths)
        # Project to Whisper d_model
        whisper_in = self.neural_to_whisper(neural_tokens)
        # Whisper encoder expects (B, T, d_model) as inputs_embeds
        whisper_out = self.whisper_encoder(inputs_embeds=whisper_in).last_hidden_state
        return self.whisper_to_llm(whisper_out)  # (B, T, llm_dim)
