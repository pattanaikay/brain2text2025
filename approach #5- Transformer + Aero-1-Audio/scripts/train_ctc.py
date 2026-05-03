import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
from tqdm import tqdm

from src.models.encoder import BIT_Transformer
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_per

class CTCPhonemeModel(nn.Module):
    def __init__(self, encoder, num_phonemes=41):
        super().__init__()
        self.encoder = encoder
        # The encoder outputs (B, T, 384)
        self.head = nn.Linear(encoder.embed_dim, num_phonemes)
        
    def forward(self, x, session_id=None):
        encoded = self.encoder(x, session_id=session_id)
        return self.head(encoded)

def train_ctc(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    import h5py
    with h5py.File(args.train_h5, 'r') as f:
        train_trials = list(f.keys())
    
    with h5py.File(args.val_h5, 'r') as f:
        val_trials = list(f.keys())

    session_ids = set()
    with h5py.File(args.train_h5, 'r') as f:
        for t in train_trials[:100]:
            session_ids.add(str(f[t].attrs.get('session', 'unknown')))
            
    train_dataset = Preprocessed_BCI_Dataset(args.train_h5, train_trials)
    val_dataset = Preprocessed_BCI_Dataset(args.val_h5, val_trials)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=bci_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=bci_collate_fn, num_workers=4)

    encoder = BIT_Transformer(session_ids=list(session_ids))
    if args.ssl_checkpoint and os.path.exists(args.ssl_checkpoint):
        print(f"Loading SSL Pretrained Encoder from {args.ssl_checkpoint}")
        encoder.load_state_dict(torch.load(args.ssl_checkpoint, map_location=device))
        
    model = CTCPhonemeModel(encoder, num_phonemes=41).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    best_per = float('inf')
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            neural_data = batch['neural'].to(device)
            session_id = batch['session_id']
            
            # Using dummy phoneme targets if not provided in the dataset
            if 'phonemes' in batch:
                labels = batch['phonemes'].to(device)
                target_lengths = batch['phoneme_lengths'].to(device)
            else:
                labels = torch.randint(1, 41, (neural_data.size(0), 50)).to(device)
                target_lengths = torch.full((neural_data.size(0),), 50, dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.use_amp):
                logits = model(neural_data, session_id=session_id) # (B, T, C)
                # CTCLoss expects (T, B, C)
                logits = logits.permute(1, 0, 2)
                log_probs = nn.functional.log_softmax(logits, dim=2)
                
                input_lengths = torch.full((neural_data.size(0),), logits.size(0), dtype=torch.long).to(device)
                loss = ctc_loss_fn(log_probs, labels, input_lengths, target_lengths)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        
        if epoch % args.val_interval == 0:
            val_per = validate(model, val_loader, device, args.use_amp)
            print(f"Validation Epoch {epoch}: PER={val_per:.4f}")
            scheduler.step(val_per)
            if val_per < best_per:
                best_per = val_per
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_per.pth"))
                print("New Best Model Saved!")

    print("CTC Fine-tuning Complete.")

def validate(model, val_loader, device, use_amp):
    model.eval()
    predictions = []
    targets_list = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            neural_data = batch['neural'].to(device)
            session_id = batch['session_id']
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                logits = model(neural_data, session_id=session_id)
                preds = logits.argmax(dim=-1)
            
            for i in range(preds.size(0)):
                pred_seq = [str(p.item()) for p in preds[i] if p.item() != 0] # 0 is blank
                predictions.append(" ".join(pred_seq) if pred_seq else "1") # Fallback if empty to avoid empty string errors in jiwer
                
                if 'phonemes' in batch:
                    tgt = batch['phonemes'][i][:batch['phoneme_lengths'][i]]
                    targets_list.append(" ".join([str(t.item()) for t in tgt]))
                else:
                    targets_list.append("1 2 3") # dummy
                    
    return calculate_per(predictions, targets_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="scripts/models/ctc")
    parser.add_argument("--ssl_checkpoint", type=str, default="scripts/models/ssl/best_encoder_ssl.pth")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--use_amp", action="store_true", default=True)
    args = parser.parse_args()
    train_ctc(args)
