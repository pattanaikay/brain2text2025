"""
tools/perplexity_test.py
------------------------
Track A2: Spoken Language Perplexity Test.

Tests whether audio-pretrained LLMs have a smaller spoken/written PPL gap
(H2: Spoken Language Prior hypothesis).

Usage:
    python tools/perplexity_test.py \\
        --spoken_file   data/bci_sentences.txt \\   # one sentence per line
        --written_file  data/wikipedia_sample.txt \\ # matched length sample
        --out           results/perplexity_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CANDIDATES = [
    ("text-only-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("text-only-7B",   "Qwen/Qwen2.5-7B-Instruct"),
    ("vision-3B",      "Qwen/Qwen2.5-VL-3B-Instruct"),
    ("audio-7B",       "Qwen/Qwen2-Audio-7B-Instruct"),
    ("phi4-vision",    "microsoft/Phi-4-multimodal-instruct"),
]


def compute_perplexity(
    model,
    tokenizer,
    sentences: list[str],
    device: torch.device,
    max_length: int = 128,
) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for sent in tqdm(sentences, desc="PPL", leave=False):
            ids = tokenizer(
                sent, return_tensors="pt", truncation=True,
                max_length=max_length, add_special_tokens=True,
            ).input_ids.to(device)
            out  = model(ids, labels=ids)
            losses.append(out.loss.item())
    return math.exp(sum(losses) / max(len(losses), 1))


def run_ppl(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spoken  = Path(args.spoken_file).read_text().strip().splitlines()
    written = Path(args.written_file).read_text().strip().splitlines()
    # Balance lengths
    n = min(len(spoken), len(written), args.max_sentences)
    spoken  = spoken[:n]
    written = written[:n]
    print(f"Evaluating on {n} spoken and {n} written sentences.")

    results = {}
    for name, model_name in CANDIDATES:
        print(f"\n{'='*60}\n  {name}: {model_name}\n{'='*60}")
        try:
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            llm = AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True,
                torch_dtype=torch.float16, device_map="auto",
            )
            ppl_spoken  = compute_perplexity(llm, tok, spoken,  device)
            ppl_written = compute_perplexity(llm, tok, written, device)
            ratio = ppl_spoken / max(ppl_written, 1e-9)

            results[name] = {
                "model": model_name,
                "ppl_spoken":  round(ppl_spoken, 3),
                "ppl_written": round(ppl_written, 3),
                "ratio_spoken_over_written": round(ratio, 4),
            }
            print(f"  PPL spoken={ppl_spoken:.2f}  written={ppl_written:.2f}  "
                  f"ratio={ratio:.4f}  (lower ratio → more spoken-adapted)")

            del llm; torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"model": model_name, "error": str(e)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nPPL results saved → {args.out}")

    print("\nRanking by ratio (lower ratio = more adapted to spoken text):")
    ranked = sorted(
        [(k, v["ratio_spoken_over_written"]) for k, v in results.items()
         if "ratio_spoken_over_written" in v],
        key=lambda x: x[1],
    )
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name}: {r:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spoken_file",   required=True)
    parser.add_argument("--written_file",  required=True)
    parser.add_argument("--out",           default="results/perplexity_results.json")
    parser.add_argument("--max_sentences", type=int, default=250)
    args = parser.parse_args()
    run_ppl(args)


if __name__ == "__main__":
    main()
