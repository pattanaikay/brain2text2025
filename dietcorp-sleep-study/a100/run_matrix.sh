#!/usr/bin/env bash
# a100/run_matrix.sh — run the full study grid on the A100, then plot.
# Gates C2/C3 (LM) on the LM existing; C0/C1 always run.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
export PYTHONIOENCODING=utf-8

CFG=${1:-configs/study.yaml}
LM=$(python3 -c "import yaml;print(yaml.safe_load(open('$CFG')).get('lm',''))")

echo "=== C0/C1 (baseline + self-label) + phoneme-inventory self-check ==="
python3 run_study.py --config "$CFG" --conditions C0 C1 --validate_mapping --out results

if [ -n "$LM" ] && [ -f "$LM" ]; then
  echo "=== C2/C3a/C3b/C4 (LM-refined, +mem anchor, +wake read, oracle) ==="
  python3 run_study.py --config "$CFG" --conditions C2 C3a C3b C4 --out results
else
  echo "LM not found ($LM) — skipping C2/C3. Run a100/build_lm.py first. Running C4 (oracle) only."
  python3 run_study.py --config "$CFG" --conditions C4 --out results
fi

echo "=== collect + plot ==="
python3 a100/collect_and_plot.py results/
echo "Done. See results/study_results.json and results/wer_vs_day.png"
