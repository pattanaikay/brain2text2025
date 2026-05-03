import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from .encoder import BIT_Transformer
from .projector import MLPProjector

class ModalityAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        # Blueprint: Must use a learnable temperature parameter (tau)
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, neural_embeds, text_embeds):
        """
        InfoNCE Loss to align neural and text embeddings.
        Args:
            neural_embeds: (Batch, Dim) - Mean-pooled neural features
            text_embeds: (Batch, Dim) - Mean-pooled text features
        """
        batch_size = neural_embeds.size(0)
        
        # Normalize to unit hypersphere
        neural_embeds = F.normalize(neural_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Cosine similarity matrix
        # Ensure temperature is strictly positive
        tau = torch.clamp(self.temperature, min=1e-4)
        logits = torch.matmul(neural_embeds, text_embeds.t()) / tau
        
        # Symmetric labels
        labels = torch.arange(batch_size, device=neural_embeds.device)
        
        loss_n = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        
        return (loss_n + loss_t) / 2

class BITModel(nn.Module):
    def __init__(self, llm_name="lmms-lab/Aero-1-Audio-1.5B", session_ids=None, quantize=True):
        super().__init__()
        self.llm_name = llm_name
        
        # 1. Load LLM with 4-bit Quantization
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
                trust_remote_code=True
            )
            self.llm = prepare_model_for_kbit_training(self.llm)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
            
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 2. Components
        self.neural_encoder = BIT_Transformer(session_ids=session_ids)
        llm_dim = self.llm.config.hidden_size
        self.projector = MLPProjector(output_dim=llm_dim)
        
        # 3. LoRA Configuration
        # Note: 'audio_projector' or 'mm_projector' depends on Aero-1-Audio's actual architecture
        # For Qwen-based multimodal models, it's often 'audio_projector' or similar.
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "audio_projector"], 
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        self.contrastive_loss_fn = ModalityAlignmentLoss()
        self.prompt = "decode the above neural activity into an English sentence:"
        
    def forward(self, neural_data, labels=None, session_id=None, return_contrastive=True):
        batch_size = neural_data.size(0)
        device = neural_data.device
        
        # 1. Neural Encoding & Projection
        neural_tokens = self.neural_encoder(neural_data, session_id=session_id) # (B, T_patch, 384)
        projected_embeds = self.projector(neural_tokens) # (B, T_patch, 1536)
        
        # 2. Prompt Embeddings
        prompt_inputs = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_inputs.input_ids)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        
        # Combine Neural + Prompt
        combined_embeds = torch.cat([projected_embeds, prompt_embeds], dim=1)
        
        loss = 0
        contrastive_loss = torch.tensor(0.0, device=device)
        
        if labels is not None:
            # 3. Text Embeddings for Labels
            label_inputs = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(device)
            label_embeds = self.llm.get_input_embeddings()(label_inputs.input_ids)
            
            # Full input for LLM (Neural + Prompt + Labels)
            full_embeds = torch.cat([combined_embeds, label_embeds], dim=1)
            
            # Masking for CE Loss: we only want to predict the labels
            # But standard CAUSAL_LM will predict everything. 
            # We can use labels parameter in self.llm(...) which handles shifting.
            # We need to set labels for non-target tokens to -100
            target_ids = label_inputs.input_ids.clone()
            # Padding prefix labels with -100
            prefix_len = combined_embeds.size(1)
            full_labels = torch.full((batch_size, prefix_len + target_ids.size(1)), -100, device=device)
            full_labels[:, prefix_len:] = target_ids
            # Also mask padding tokens in labels
            full_labels[full_labels == self.tokenizer.pad_token_id] = -100
            
            attention_mask = torch.ones(full_embeds.shape[:2], device=device)
            
            outputs = self.llm(inputs_embeds=full_embeds, attention_mask=attention_mask, labels=full_labels)
            ce_loss = outputs.loss
            
            # 4. Contrastive Alignment Loss
            if return_contrastive:
                # Mean pool neural tokens (before projector or after?)
                # Usually after projector to align with LLM space
                neural_pooled = projected_embeds.mean(dim=1)
                
                # Mean pool text tokens (from labels)
                # Mask out padding before pooling
                text_mask = (label_inputs.input_ids != self.tokenizer.pad_token_id).unsqueeze(-1)
                text_pooled = (label_embeds * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1)
                
                contrastive_loss = self.contrastive_loss_fn(neural_pooled, text_pooled)
                
            loss = ce_loss + contrastive_loss
            return loss, ce_loss, contrastive_loss
            
        return combined_embeds

    def generate(self, neural_data, session_id=None, max_new_tokens=100, top_p=0.9, temperature=0.7):
        self.eval()
        with torch.no_grad():
            neural_tokens = self.neural_encoder(neural_data, session_id=session_id)
            projected_embeds = self.projector(neural_tokens)
            
            prompt_inputs = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False).to(neural_data.device)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_inputs.input_ids)
            prompt_embeds = prompt_embeds.repeat(neural_data.size(0), 1, 1)
            
            combined_embeds = torch.cat([projected_embeds, prompt_embeds], dim=1)
            
            outputs = self.llm.generate(
                inputs_embeds=combined_embeds,
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
