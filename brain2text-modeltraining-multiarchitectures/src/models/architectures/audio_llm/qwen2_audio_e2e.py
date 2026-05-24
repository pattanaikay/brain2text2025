"""Arch-5a: Qwen2-Audio-7B-Instruct as the LLM decoder.

The BIT neural encoder feeds inputs_embeds directly into the text decoder,
bypassing Qwen2-Audio's audio branch entirely.  llm_dim = 3584.
Requires: pip install transformers>=4.37 peft bitsandbytes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

from src.models.encoder import BIT_Transformer
from src.models.projector import MLPProjector

LLM_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

PROMPT_START = "<|im_start|>user\n<|audio_bos|>"
PROMPT_END = "<|audio_eos|>\nWhat is the speaker saying?<|im_end|>\n<|im_start|>assistant\n"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class BrainToTextQwen2Audio(nn.Module):
    """BIT encoder + Qwen2-Audio-7B-Instruct decoder.

    Memory: batch_size=2, accumulation_steps=16 on A100-40GB with 4-bit NF4.
    """

    def __init__(self, session_ids=None, quantize=True, patch_size=4):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kw = dict(
            quantization_config=bnb_config if quantize else None,
            torch_dtype=None if quantize else torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, **load_kw)
        if quantize:
            self.llm = prepare_model_for_kbit_training(self.llm)

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # text_config.hidden_size for multimodal Qwen2-Audio = 3584
        if hasattr(self.llm.config, "text_config"):
            llm_dim = self.llm.config.text_config.hidden_size
        else:
            llm_dim = self.llm.config.hidden_size

        self.neural_encoder = BIT_Transformer(session_ids=session_ids, patch_size=patch_size)
        self.projector = MLPProjector(output_dim=llm_dim)

        lora_cfg = LoraConfig(
            r=8, lora_alpha=32,
            target_modules=LORA_TARGETS,
            lora_dropout=0.2, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        self.ctc_head = nn.Linear(384, 42)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.prompt_start = PROMPT_START
        self.prompt_end = PROMPT_END
