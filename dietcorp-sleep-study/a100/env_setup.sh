#!/usr/bin/env bash
# a100/env_setup.sh — set up the study environment on a fresh A100 (Linux).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Torch (CUDA 12.1) first, then the rest
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Optional: KenLM for faster/larger n-gram LM (pure-Python fallback works without it)
pip install kenlm || echo "kenlm not installed — using pure-Python n-gram fallback"

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "env ready. Next: python a100/prepare_data.py ; python a100/build_lm.py ; bash a100/run_matrix.sh"
