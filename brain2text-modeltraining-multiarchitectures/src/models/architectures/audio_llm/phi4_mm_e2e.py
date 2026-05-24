"""Arch-5c: Phi-4-multimodal-instruct as the LLM decoder.

Requires: pip install transformers>=4.40 (trust_remote_code=True)
Hidden size: 3072.  LoRA targets: qkv_proj, o_proj, gate_up_proj, down_proj.
"""

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from src.models.encoder import BIT_Transformer
from src.models.projector import MLPProjector

LLM_NAME = "microsoft/Phi-4-multimodal-instruct"

# Phi-4 uses fused QKV; confirm by print(model) before changing
LORA_TARGETS = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

PROMPT_START = "<|user|>\n<|audio_1|>\n"
PROMPT_END = "<|end|><|assistant|>\n"


class BrainToTextPhi4MM(nn.Module):
    """BIT encoder + Phi-4-multimodal decoder.

    llm_dim = 3072.  batch_size=2, accumulation_steps=16 on A100-40GB.
    """

    def __init__(self, session_ids=None, quantize=True, patch_size=4):
        super().__init__()
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        ) if quantize else None

        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            quantization_config=bnb_cfg,
            torch_dtype=None if quantize else torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if quantize:
            self.llm = prepare_model_for_kbit_training(self.llm)

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_dim = self.llm.config.hidden_size  # 3072

        self.neural_encoder = BIT_Transformer(session_ids=session_ids, patch_size=patch_size)
        self.projector = MLPProjector(output_dim=llm_dim)

        lora_cfg = LoraConfig(
            r=8, lora_alpha=32, target_modules=LORA_TARGETS,
            lora_dropout=0.2, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        self.ctc_head = nn.Linear(384, 42)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.prompt_start = PROMPT_START
        self.prompt_end = PROMPT_END
