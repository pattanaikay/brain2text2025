import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path so we can import src
base_path = Path(__file__).parent.parent
sys.path.insert(0, str(base_path))

from src.models.baseline import BITModel
import time

def dry_run():
    print("🚀 Starting Pre-flight Dry Run...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Initialize Model
    # Note: We use a small mock LLM configuration or dummy weights if the real one isn't available
    # But for a true audit, we attempt to initialize the real BITModel architecture.
    print("Initializing BITModel (Encoder + Projector + QLoRA LLM)...")
    try:
        # For dry run, we might want to skip heavy quantization if on a low-memory CPU
        # But BITModel defaults are what will be used in the cluster.
        model = BITModel(session_ids=["session_test_1", "session_test_2"], quantize=torch.cuda.is_available()).to(device)
        print("✅ Model Initialization Successful.")
    except Exception as e:
        print(f"❌ Model Initialization Failed: {e}")
        return

    # 2. Generate Synthetic Batch
    print("Generating synthetic batch [Batch=4, Time=500, Features=512]...")
    batch_size = 4
    time_steps = 500
    features = 512
    
    neural_spikes = torch.randn(batch_size, time_steps, features).to(device)
    mock_labels = [
        "the quick brown fox jumps over the lazy dog",
        "brain to text integration transformer is active",
        "testing neural decoding pipeline",
        "sample sentence for dry run"
    ]
    session_id = ["session_test_1"] * batch_size

    # 3. Forward Pass
    print("Executing Forward Pass...")
    try:
        start_time = time.time()
        loss, ce_loss, contrastive_loss = model(neural_spikes, labels=mock_labels, session_id=session_id)
        end_time = time.time()
        
        print(f"✅ Forward Pass Successful ({end_time - start_time:.2f}s).")
        print(f"   - Total Loss: {loss.item():.4f}")
        print(f"   - CE Loss: {ce_loss.item():.4f}")
        print(f"   - Contrastive Loss: {contrastive_loss.item():.4f}")
    except Exception as e:
        print(f"❌ Forward Pass Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Backward Pass
    print("Executing Backward Pass & Optimizer Step...")
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("✅ Backward Pass & Optimizer Step Successful.")
    except Exception as e:
        print(f"❌ Backward Pass Failed: {e}")
        return

    # 5. Final Report
    print("\n" + "="*40)
    print("FINAL AUDIT REPORT: SUCCESS")
    print("="*40)
    if torch.cuda.is_available():
        print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Max Memory Cached: {torch.cuda.max_memory_reserved() / 1024**2:.2f} MB")
    
    # Verify Gradient Requirements
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters (LoRA + Encoder + Projector): {trainable_params:,}")
    
    # Shape verification
    neural_tokens = model.neural_encoder(neural_spikes, session_id=session_id)
    print(f"Neural Encoder Output Shape: {neural_tokens.shape} (Expected [4, 100, 384])")
    
    print("\nPre-flight check complete. Ready for deployment to JarvisLabs.ai.")
    print("="*40)

if __name__ == "__main__":
    dry_run()
