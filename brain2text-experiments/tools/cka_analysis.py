"""
tools/cka_analysis.py
---------------------
Track A1: CKA Embedding Space Analysis.

For each candidate LLM, compute linear CKA between:
  - BIT encoder + projector outputs (neural_pooled)
  - LLM embedding layer outputs on ground-truth text (text_pooled)

High CKA → H3 is true (embedding geometry matters).
Run locally, no training, ~30 min on RTX 4050.

Usage:
    python tools/cka_analysis.py \\
        --val_h5   data/val.hdf5 \\
        --ckpt     ../brain2text-modeltraining/outputs/ctc/best_model_per.pth \\
        --out      results/cka_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from docks.bit_dock import BIT_Transformer, MLPProjector
from docks.multiarch_dock import Preprocessed_BCI_Dataset, bci_collate_fn


CANDIDATES = [
    ("text-only-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("text-only-7B",   "Qwen/Qwen2.5-7B-Instruct"),
    ("vision-3B",      "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("audio-7B",       "Qwen/Qwen2-Audio-7B-Instruct"),
    ("phi4-vision",    "microsoft/Phi-4-multimodal-instruct"),
]


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Linear CKA. X, Y: (N, d) float tensors.
    Returns a scalar in [0, 1]; higher = more aligned.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y
    num   = (XtY * XtY).sum()
    denom = ((XtX * XtX).sum() * (YtY * YtY).sum()).sqrt()
    return (num / denom.clamp(min=1e-8)).item()


def _random_proj(X: torch.Tensor, out_dim: int, seed: int = 0) -> torch.Tensor:
    """Project X (N, d_in) → (N, out_dim) with a fixed random orthogonal map."""
    torch.manual_seed(seed)
    W = torch.randn(X.size(1), out_dim, device=X.device)
    W, _ = torch.linalg.qr(W)
    return X @ W


def collect_neural_embeddings(
    encoder: BIT_Transformer,
    projector: MLPProjector,
    val_loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Run BIT+projector on val set, mean-pool → (N_val, llm_dim)."""
    encoder.eval(); projector.eval()
    all_pooled = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Neural embeddings"):
            neural  = batch["neural"].to(device)
            lengths = batch["neural_lengths"].to(device)
            sid     = batch["session_id"]
            tokens  = encoder(neural, session_id=sid, neural_lengths=lengths)
            proj    = projector(tokens)                    # (B, T, llm_dim)
            # mean-pool ignoring padding
            patch_size = encoder.patch_size
            plen = (lengths + patch_size - 1) // patch_size
            pooled = []
            for i in range(proj.size(0)):
                pooled.append(proj[i, :max(1, plen[i].item())].mean(0))
            all_pooled.append(torch.stack(pooled).cpu())
    return torch.cat(all_pooled, 0)   # (N_val, llm_dim)


def collect_text_embeddings(
    model_name: str,
    tokenizer,
    llm,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Embed ground-truth sentences and mean-pool → (N_val, llm_hidden)."""
    emb_layer = llm.get_input_embeddings()
    all_pooled = []
    with torch.no_grad():
        for text in tqdm(texts, desc=f"Text embeddings [{model_name}]"):
            ids  = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
            embs = emb_layer(ids.input_ids)        # (1, T, llm_hidden)
            all_pooled.append(embs.squeeze(0).mean(0).cpu())
    return torch.stack(all_pooled)   # (N_val, llm_hidden)


def run_cka(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Detect embedding dimension from checkpoint ──────────────────
    embed_dim = 2048  # default
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        # Extract embedding dimension from checkpoint
        for key, val in sd.items():
            if "patch_embedding.weight" in key:
                embed_dim = val.shape[1]
                print(f"Detected embed_dim={embed_dim} from checkpoint")
                break

    # ── Load encoder ──────────────────────────────────────────────────
    encoder   = BIT_Transformer(patch_size=args.patch_size, embed_dim=embed_dim).to(device)
    projector = MLPProjector().to(device)   # output_dim=1536 default

    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        sd   = ckpt.get("model_state_dict", ckpt)
        enc_sd = {k.replace("encoder.","",1): v for k,v in sd.items() if not k.startswith("head.")}
        encoder.load_state_dict(enc_sd, strict=False)
        print(f"Loaded encoder from {args.ckpt}")

    # ── Load val data ─────────────────────────────────────────────────
    import glob, os
    h5s = ([args.val_h5] if os.path.isfile(args.val_h5)
           else sorted(glob.glob(os.path.join(args.val_h5,"**/data_val.hdf5"), recursive=True)))
    dataset = Preprocessed_BCI_Dataset(h5s, patch_size=args.patch_size, augment=False)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                          collate_fn=bci_collate_fn, num_workers=2)
    all_texts = [dataset[i]["text"] for i in range(len(dataset))]

    # ── Collect neural pooled embeddings ──────────────────────────────
    # Use 1536-dim (Qwen2.5-1.5B) as the neural embedding dim;
    # for other LLMs with different dims we project randomly for CKA.
    neural_pooled = collect_neural_embeddings(encoder, projector, loader, device)
    print(f"Neural pooled: {neural_pooled.shape}")

    results = {}

    for name, model_name in CANDIDATES:
        print(f"\n{'='*60}\n  {name}: {model_name}\n{'='*60}")
        try:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # No quantization — preserve embedding geometry
            llm = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float32, device_map="cpu",
            ).eval()
            llm_dim = (llm.config.text_config.hidden_size
                       if hasattr(llm.config, "text_config")
                       else llm.config.hidden_size)

            text_pooled = collect_text_embeddings(name, tok, llm, all_texts, torch.device("cpu"))
            # Move neural_pooled to same device, match dims if needed
            X = neural_pooled.float()
            Y = text_pooled.float()
            if X.size(1) != Y.size(1):
                X = _random_proj(X, Y.size(1))

            cka_val = linear_cka(X, Y)
            results[name] = {"model": model_name, "llm_dim": llm_dim, "cka": round(cka_val, 6)}
            print(f"  CKA = {cka_val:.6f}")

            # Free GPU/CPU memory between models
            del llm; torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"model": model_name, "error": str(e)}

    # ── Save ──────────────────────────────────────────────────────────
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCKA results saved → {args.out}")
    print("\nRanking (higher CKA = better aligned):")
    ranked = sorted(
        [(k, v["cka"]) for k, v in results.items() if "cka" in v],
        key=lambda x: x[1], reverse=True,
    )
    for i, (name, cka) in enumerate(ranked, 1):
        print(f"  {i}. {name}: {cka:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_h5",     required=True)
    parser.add_argument("--ckpt",       default=None)
    parser.add_argument("--out",        default="results/cka_results.json")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    run_cka(args)


if __name__ == "__main__":
    main()
