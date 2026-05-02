import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_training_curves(history_file_path, output_path='outputs/training_curves.png'):
    """
    Reads training history and generates a 4-panel matplotlib dashboard.
    """
    if not os.path.exists(history_file_path):
        print(f"Error: Could not find {history_file_path}")
        return

    # Load metrics
    with open(history_file_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    # Initialize a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Approach 5: End-to-End BIT Training Dynamics', fontsize=16)

    # 1. Loss (Cross Entropy + Contrastive)
    axs[0, 0].plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    if 'val_loss' in history and history['val_loss']:
        val_epochs = [i for i, v in enumerate(history['val_loss'], 1) if v > 0]
        val_values = [v for v in history['val_loss'] if v > 0]
        if val_values:
            axs[0, 0].plot(val_epochs, val_values, label='Val Loss', color='orange', marker='o')
    axs[0, 0].set_title('Training vs Validation Loss')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Character Error Rate (CER)
    if 'train_cer' in history and history['train_cer']:
        axs[0, 1].plot(epochs, history['train_cer'], label='Train CER', color='blue')
    if 'val_cer' in history and history['val_cer']:
        val_epochs = [i for i, v in enumerate(history['val_cer'], 1) if v < 1.0]
        val_values = [v for v in history['val_cer'] if v < 1.0]
        if val_values:
            axs[0, 1].plot(val_epochs, val_values, label='Val CER', color='orange', marker='o')
    axs[0, 1].set_title('Character Error Rate (CER)')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('CER')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Word Error Rate (WER)
    if 'train_wer' in history and history['train_wer']:
        axs[1, 0].plot(epochs, history['train_wer'], label='Train WER', color='blue')
    if 'val_wer' in history and history['val_wer']:
        val_epochs = [i for i, v in enumerate(history['val_wer'], 1) if v < 1.0]
        val_values = [v for v in history['val_wer'] if v < 1.0]
        if val_values:
            axs[1, 0].plot(val_epochs, val_values, label='Val WER', color='orange', marker='o')
    axs[1, 0].set_title('Word Error Rate (WER)')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('WER')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Intelligence Gap (Val CER vs Val WER)
    if 'val_cer' in history and history['val_wer']:
        val_epochs = [i for i, v in enumerate(history['val_wer'], 1) if v < 1.0]
        val_cer = [v for i, v in enumerate(history['val_cer'], 1) if history['val_wer'][i-1] < 1.0]
        val_wer = [v for v in history['val_wer'] if v < 1.0]
        if val_cer and val_wer:
            axs[1, 1].plot(val_epochs, val_cer, label='Val CER (Acoustic)', color='blue', marker='s')
            axs[1, 1].plot(val_epochs, val_wer, label='Val WER (Total Pipeline)', color='red', marker='D')
    axs[1, 1].set_title('The Intelligence Gap (Val CER vs Val WER)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Error Rate')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Save and show
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for main title
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Metrics dashboard saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="outputs/training_history.json")
    parser.add_argument("--output", type=str, default="outputs/training_curves.png")
    args = parser.parse_args()
    plot_training_curves(args.history, args.output)
