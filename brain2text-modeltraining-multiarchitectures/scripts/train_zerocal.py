"""Zero-calibration inference for ZenBrain (Arch-4).

Loads a pretrained ZenBrain checkpoint, freezes all weights, enables the
episodic buffer, and runs validation trials sequentially.  After each trial
the buffer is updated with high-confidence patch tokens from the CTC head.
WER is reported per-trial to show adaptation as the buffer fills.

Usage:
    python scripts/train_zerocal.py \\
        --checkpoint outputs/e2e_zenbrain/best_model_wer.pth \\
        --val_h5 data/ --output_dir outputs/zerocal_zenbrain
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.models.bit_e2e import BrainToTextE2E
from src.preprocessing.dataloader import Preprocessed_BCI_Dataset, bci_collate_fn
from src.utils.metrics import calculate_wer, calculate_cer
from src.utils.logging_utils import setup_logging


def run_zerocal(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, log_name="zerocal")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    import glob
    if os.path.isdir(args.val_h5):
        val_files = sorted(glob.glob(os.path.join(args.val_h5, "**/data_val.hdf5"), recursive=True))
    else:
        val_files = [args.val_h5]

    val_dataset = Preprocessed_BCI_Dataset(val_files, patch_size=args.patch_size, augment=False)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=bci_collate_fn, num_workers=0,
    )

    # Load model in no-quantize mode for inference flexibility
    model = BrainToTextE2E(quantize=False, patch_size=args.patch_size, encoder_name="zenbrain").to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state if not isinstance(state, dict) or "model_state_dict" not in state
                          else state["model_state_dict"], strict=False)

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Enable episodic buffer
    model.neural_encoder.use_memory = True
    logger.info("ZeroCal inference: all weights frozen, episodic buffer enabled.")

    wer_curve = []
    model.eval()
    for trial_idx, batch in enumerate(val_loader):
        neural = batch["neural"].to(device)
        lengths = batch["neural_lengths"].to(device)
        session_id = batch["session_id"]
        label = batch["text"]

        with torch.no_grad():
            # Encoder forward to get tokens and CTC logits for buffer update
            neural_tokens = model.neural_encoder(neural, session_id=session_id,
                                                  neural_lengths=lengths)
            ctc_logits = model.ctc_head(neural_tokens.float())
            model.neural_encoder.update_buffer_from_outputs(neural_tokens, ctc_logits)

            preds = model.generate(neural, session_id=session_id, neural_lengths=lengths)

        wer = calculate_wer(label, preds)
        wer_curve.append(float(wer))
        if trial_idx % 50 == 0:
            logger.info(f"Trial {trial_idx}: WER={wer:.4f}  pred='{preds[0][:60]}'")

    avg_wer = sum(wer_curve) / max(1, len(wer_curve))
    logger.info(f"Zero-cal final avg WER: {avg_wer:.4f} over {len(wer_curve)} trials")

    with open(os.path.join(args.output_dir, "zerocal_wer_curve.json"), "w") as f:
        json.dump({"wer_curve": wer_curve, "avg_wer": avg_wer}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_h5", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/zerocal_zenbrain")
    parser.add_argument("--patch_size", type=int, default=4)
    args = parser.parse_args()
    run_zerocal(args)
