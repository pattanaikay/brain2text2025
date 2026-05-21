# Program: ECoG HRM / NeuroMoE Autoresearch

You are running short, metric-driven architecture experiments on ECoG finger-flexion regression.

Primary metric: validation mean Pearson correlation across five finger-flexion targets. Higher is better.

Secondary metrics: RMSE, runtime, parameter count, peak VRAM, expert utilization, router entropy.

Only edit `train.py`. The benchmark API must remain compatible with:

```python
train_model(config: TrainConfig) -> dict
```

A valid result dictionary includes:

- `mean_pearson`
- `rmse`
- `params`
- `epochs`
- `steps`
- `train_seconds`
- `peak_vram_mb`
- `history`
- `predictions_path`

Default run:

```powershell
python benchmark.py --model auto --budget-minutes 5 --seed 13
```

Keep changes that improve validation `mean_pearson` with no benchmark failure.
