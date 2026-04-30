import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

class NeuralEncoder(nn.Module):
    def __init__(self, input_dim=512, embed_dim=384, num_layers=7, num_heads=6, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=1024,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (batch, time, 512)
        x = self.input_proj(x)
        x = self.transformer_encoder(x)
        x = self.layer_norm(x)
        return x

class MLPProjector(nn.Module):
    def __init__(self, embed_dim=384, llm_dim=1536):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, llm_dim),
            nn.ReLU(),
            nn.Linear(llm_dim, llm_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

class BrainToTextModel(nn.Module):
    def __init__(self, llm_name="lmms-lab/Aero-1-Audio-1.5B", quantize=True):
        super().__init__()
        self.llm_name = llm_name
        
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
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name, trust_remote_code=True)
            
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.neural_encoder = NeuralEncoder()
        llm_dim = self.llm.config.hidden_size
        self.projector = MLPProjector(llm_dim=llm_dim)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, lora_config)
        
        self.prompt = "decode the above neural activity into an English sentence:"
        
    def forward(self, neural_data, labels=None):
        # Encode & Project
        neural_embeds = self.neural_encoder(neural_data)
        projected_embeds = self.projector(neural_embeds)
        
        # Prompt Embeddings
        prompt_inputs = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False).to(neural_data.device)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_inputs.input_ids)
        
        batch_size = neural_data.size(0)
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        
        combined_embeds = torch.cat([prompt_embeds, projected_embeds], dim=1)
        
        if labels is not None:
            label_inputs = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(neural_data.device)
            label_embeds = self.llm.get_input_embeddings()(label_inputs.input_ids)
            
            full_embeds = torch.cat([combined_embeds, label_embeds], dim=1)
            attention_mask = torch.ones(full_embeds.shape[:2], device=neural_data.device)
            
            outputs = self.llm(inputs_embeds=full_embeds, attention_mask=attention_mask, labels=label_inputs.input_ids)
            return outputs.loss
            
        return combined_embeds

    def generate(self, neural_data, max_new_tokens=100, top_p=0.9, temperature=0.7):
        self.llm.eval()
        with torch.no_grad():
            neural_embeds = self.neural_encoder(neural_data)
            projected_embeds = self.projector(neural_embeds)
            
            prompt_inputs = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=False).to(neural_data.device)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_inputs.input_ids)
            
            batch_size = neural_data.size(0)
            prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
            
            combined_embeds = torch.cat([prompt_embeds, projected_embeds], dim=1)
            
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
            for i in range(batch_size):
                start_idx = prompt_inputs.input_ids.shape[1] + projected_embeds.shape[1]
                gen_ids = outputs[i][start_idx:]
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                generated_texts.append(text)
                
        return generated_texts
