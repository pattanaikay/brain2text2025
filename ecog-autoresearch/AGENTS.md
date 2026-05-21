# Autoresearch Agent Instructions

Goal: improve mean validation Pearson correlation for ECoG finger-flexion regression under a 6 GB VRAM budget.

Editable file:

- `train.py`

Read-only harness files:

- `prepare_data.py`
- `benchmark.py`
- `plot_results.py`
- `results.tsv` schema

Benchmark command:

```powershell
python benchmark.py --model auto --budget-minutes 5
```

Loop protocol:

1. Inspect `results.tsv` and the latest run artifacts.
2. Make exactly one architecture or training change in `train.py`.
3. Run the benchmark with the same budget and seed unless intentionally testing robustness.
4. Keep the change only if `mean_pearson` improves and the run completes.
5. Record notes in the benchmark `--notes` argument.

Promising experiment families:

- Replace or improve the compact CNN baseline.
- Add RoPE/local attention refinements to the tiny Transformer.
- Improve NeuroMoE expert specialization without router collapse.
- Tune HRM low/high cycle counts for the same wall-clock budget.
- Combine HRM recurrence with MoE only after each helps independently.

Do not:

- Change the dataset split while comparing architectures.
- Optimize for training loss alone.
- Increase model size until it no longer fits a 6 GB GPU.
- Add a large LLM decoder before the neural encoder benchmark is stable.
