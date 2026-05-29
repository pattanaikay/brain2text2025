"""
tools/phoneme_probe.py
----------------------
Track A3: Phoneme Probing Accuracy.

Trains a linear probe (Linear(llm_dim, 42)) on frozen LLM hidden states
to predict phoneme labels. Tests H1 (Phoneme Prior hypothesis).

Audio-pretrained LLMs should have significantly higher probe accuracy.

Usage:
    python tools/phoneme_probe.py \\
        --val_h5   data/val.hdf5 \\
        --out      results/phoneme_probe_results.json \\
        --epochs   20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docks.multiarch_dock import Preprocessed_BCI_Dataset, bci_collate_fn

CANDIDATES = [
    ("text-only-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("audio-7B",       "Qwen/Qwen2-Audio-7B-Instruct"),
    ("vision-3B",      "Qwen/Qwen2.5-VL-3B-Instruct"),
]
N_PHONEMES = 42


def collect_hidden_states(
    llm, tokenizer, texts: list[str], device: torch.device,
) -> torch.Tensor:
    """
    Embed ground-truth sentences; return mean-pooled hidden states
    from the last transformer layer → (N_val, llm_hidden).
    """
    llm.eval()
    all_hs = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Collect hidden states", leave=False):
            ids = tokenizer(text, return_tensors="pt",
                            add_special_tokens=True).input_ids.to(device)
            out = llm(ids, output_hidden_states=True)
            # Last layer hidden state, mean-pool over tokens
            hs  = out.hidden_states[-1].squeeze(0).mean(0)  # (llm_hidden,)
            all_hs.append(hs.cpu())
    return torch.stack(all_hs)   # (N, llm_hidden)


def collect_phoneme_labels(val_loader: DataLoader) -> torch.Tensor:
    """
    Collect first-phoneme label per sample (majority vote over the sequence).
    Returns (N,) int64 tensor with values in [0, 41].
    """
    all_labels = []
    for batch in val_loader:
        phonemes = batch.get("phonemes")   # (B, T_phon) or None
        if phonemes is None:
            raise ValueError(
                "Dataset does not have phoneme labels. "
                "Ensure CTC-training data is used for A3."
            )
        # Majority phoneme in each sequence as a proxy label
        for b in range(phonemes.size(0)):
            seq = phonemes[b][phonemes[b] > 0]   # exclude blank=0
            if len(seq) == 0:
                all_labels.append(1)
            else:
                mode = torch.mode(seq).values.item()
                all_labels.append(int(mode))
    return torch.tensor(all_labels, dtype=torch.long)


def train_probe(
    X: torch.Tensor,    # (N, llm_dim) float
    y: torch.Tensor,    # (N,) int64
    n_classes: int = N_PHONEMES,
    epochs:    int = 20,
    lr:        float = 1e-3,
    device:    torch.device = torch.device("cpu"),
) -> dict:
    X, y  = X.to(device), y.to(device)
    probe = nn.Linear(X.size(1), n_classes).to(device)
    opt   = torch.optim.Adam(probe.parameters(), lr=lr)
    ds    = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    for ep in range(epochs):
        probe.train()
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(probe(xb), yb).backward()
            opt.step()

    # Eval
    probe.eval()
    with torch.no_grad():
        logits = probe(X)
        top1   = (logits.argmax(1) == y).float().mean().item()
        top5   = (y.unsqueeze(1) == logits.topk(5, dim=1).indices).any(1).float().mean().item()
    return {"top1_acc": round(top1, 4), "top5_acc": round(top5, 4)}


def run_probe(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import glob, os
    h5s = ([args.val_h5] if os.path.isfile(args.val_h5)
           else sorted(glob.glob(os.path.join(args.val_h5,"**/data_val.hdf5"),recursive=True)))
    dataset = Preprocessed_BCI_Dataset(h5s, patch_size=4, augment=False)
    loader  = DataLoader(dataset, batch_size=16, shuffle=False,
                          collate_fn=bci_collate_fn, num_workers=2)
    texts   = [dataset[i]["text"] for i in range(len(dataset))]

    phoneme_labels = collect_phoneme_labels(loader)

    results = {}
    for name, model_name in CANDIDATES:
        print(f"\n{'='*60}\n  {name}: {model_name}\n{'='*60}")
        try:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            llm = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto",
            )
            hidden = collect_hidden_states(llm, tok, texts, device)
            acc    = train_probe(hidden, phoneme_labels, epochs=args.epochs, device=device)
            results[name] = {"model": model_name, **acc}
            print(f"  Top-1={acc['top1_acc']:.4f}  Top-5={acc['top5_acc']:.4f}")
            del llm; torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"model": model_name, "error": str(e)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPhoneme probe results saved → {args.out}")

    print("\nRanking (higher top-1 = more phoneme info in LLM hidden states):")
    ranked = sorted(
        [(k, v["top1_acc"]) for k, v in results.items() if "top1_acc" in v],
        key=lambda x: x[1], reverse=True,
    )
    for i, (name, acc) in enumerate(ranked, 1):
        print(f"  {i}. {name}: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_h5",  required=True)
    parser.add_argument("--out",     default="results/phoneme_probe_results.json")
    parser.add_argument("--epochs",  type=int, default=20)
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
