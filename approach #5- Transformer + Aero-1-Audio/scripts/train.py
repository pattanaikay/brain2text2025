import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import time
from tqdm import tqdm

from src.models.baseline import BrainToTextModel
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_wer, calculate_cer

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BrainToTextModel(quantize=True).to(self.device)
        self.tokenizer = self.model.tokenizer
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)
        
        self.best_wer = float('inf')
        self.patience_counter = 0
        
    def ssl_pretrain(self, train_loader, epochs=50):
        print("Starting SSL Pretraining...")
        self.model.neural_encoder.train()
        self.model.projector.eval()
        self.model.llm.eval()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader):
                neural_data = batch['neural'].to(self.device)
                # Masked modeling: randomly mask some time steps
                mask = torch.rand_like(neural_data) < 0.15
                masked_data = neural_data.clone()
                masked_data[mask] = 0
                
                self.optimizer.zero_grad()
                # Forward pass through encoder only
                encoded = self.model.neural_encoder(masked_data)
                target_encoded = self.model.neural_encoder(neural_data)
                
                # MSE Loss on masked positions
                loss = nn.MSELoss()(encoded[mask], target_encoded[mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            print(f"SSL Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            neural_data = batch['neural'].to(self.device)
            labels = batch['text']
            
            self.optimizer.zero_grad()
            loss = self.model(neural_data, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
        
    def validate(self, val_loader):
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                neural_data = batch['neural'].to(self.device)
                preds = self.model.generate(neural_data)
                predictions.extend(preds)
                targets.extend(batch['text'])
                
        wer = calculate_wer(predictions, targets)
        cer = calculate_cer(predictions, targets)
        return wer, cer
        
    def train(self, train_loader, val_loader, epochs=500):
        # SSL Phase
        self.ssl_pretrain(train_loader, epochs=50)
        
        # Fine-tuning Phase
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_epoch(train_loader, epoch)
            self.scheduler.step(train_loss)
            
            if epoch % 10 == 0:
                wer, cer = self.validate(val_loader)
                print(f"Val WER: {wer:.4f}, Val CER: {cer:.4f}")
                
                if wer < self.best_wer:
                    self.best_wer = wer
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), self.config.output_dir + "/best_model_wer.pth")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                    
            if epoch % 50 == 0:
                torch.save(self.model.state_dict(), self.config.output_dir + f"/checkpoint_epoch_{epoch}.pth")
                
        print("Training complete.")
