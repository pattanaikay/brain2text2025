# Step 1 (once): make toy dataset
python tools/make_toy_dataset.py --full_path data/data_train.hdf5 --toy_path data/toy_train.hdf5

# Step 2: run shape tests (no GPU needed)
python -m pytest tests/test_stage_shapes.py -v -m "not slow"

# Step 3: first toy experiment (B0 baseline)
python run.py --expt B0_baseline --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

# Step 4: Track A analysis (no training, ~30 min)
python tools/cka_analysis.py --val_h5 data/val.hdf5 --out results/cka_results.json

# Step 5: compare results
python results/leaderboard.py --list
python results/plot_tracks.py --track B --profile toy

Here's the complete picture from `run.py`, the profiles, and the registry.

---

## The mandatory three-step progression

Every experiment must follow this order — `run.py` enforces it:

```
pytest (CPU) → toy run (local GPU, ~20 min) → full run (A100, 150 epochs)python tools/make_toy_dataset.py --full_path data/data_train.hdf5 --toy_path data/toy_train.hdf5
```

The full run **hard-blocks** if no toy pass exists in `leaderboard.sqlite` within the last 7 days.

---

## Step 1 — Shape gate (CPU, ~10 sec, no data needed)

```bash
cd C:\Projects\Brain2Text2025\brain2text2025\brain2text-experiments
python -m pytest tests/test_stage_shapes.py -v
```

Run this once before touching anything else. It catches wrong tensor shapes in every encoder/projector without touching a GPU.

---

## Step 2 — Toy runs (local RTX 4050, ~20 min each)

**Basic pattern:**
```bash
python run.py --expt <ID> --profile toy \
    --train_h5 data/toy_train.hdf5 \
    --val_h5   data/val.hdf5
```

### Track B — Encoder variants (run in dependency order)

```bash
# B0 first — all B1-B5 compare against this baseline
python run.py --expt B0_baseline --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

# B1-B5 can run in any order after B0
python run.py --expt B1          --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
python run.py --expt B2          --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
python run.py --expt B3          --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # GRU fallback
python run.py --expt B3_mamba    --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # needs mamba-ssm
python run.py --expt B4          --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
python run.py --expt B5          --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5
```

> **B3_mamba** requires `pip install mamba-ssm causal-conv1d` and CUDA 11.8+. Run B3 (GRU) first to confirm the architecture is sound before installing the C extension.

### Track C — Decoder variants

```bash
# C3 runs locally (~4.1 GB)
python run.py --expt C3 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

# C1 and C2 are 7B/5.6B models — exceed 6 GB locally; run on cloud only
# python run.py --expt C1 --profile toy ...   ← skip locally
# python run.py --expt C2 --profile toy ...   ← skip locally
```

### Track D — Loss ablations (all run locally, same architecture)

```bash
# Each D experiment is independent — run in any order
python run.py --expt D1b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # CTC linear anneal
python run.py --expt D1d --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # no CTC
python run.py --expt D2a --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # no contrastive
python run.py --expt D2d --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # contrastive ×2
python run.py --expt D3b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # topo 0.001
python run.py --expt D3c --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # topo 0.01
python run.py --expt D4  --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # label smoothing
```

### Track E — Projector variants

```bash
python run.py --expt E1a --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # deep MLP
python run.py --expt E1b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # gated MLP

# E2b first (E3 depends on it)
python run.py --expt E2b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # Q-Former 32

# E3 is a 6-cell grid search (patch_size × n_queries) — run.py iterates the grid internally
python run.py --expt E3  --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5  # ~2 hrs total
```

---

## Step 3 — Full runs (A100 cloud, 150 epochs)

Only after toy passes. `run.py` checks `leaderboard.sqlite` automatically.

```bash
python run.py --expt B1 --profile full \
    --train_h5 /data/brain2text/ \
    --val_h5   /data/brain2text/
```

After the run finishes the JarvisLabs instance auto-pauses (controlled by `profiles/full.yaml → auto_pause: true`).

---

## What happens automatically during each run

| Step | What `run.py` does |
|---|---|
| **Step 50** | `SmokeAssert` checks CE < 10, CTC < 5, no NaNs, throughput ≥ 50 tok/s — crashes fast if broken |
| **Epoch 10** | Records `WER@10` to `leaderboard.sqlite` (primary ranking metric) |
| **Every 2 epochs** (toy) | Validation → WER, CER, loss logged; `ReduceLROnPlateau` steps |
| **Best checkpoint** | `best_encoder.pth` + `best_projector.pth` saved on WER improvement |
| **End of run** | Computes convergence slope, writes full record to `leaderboard.sqlite` |

---

## Viewing results after runs

```bash
# See leaderboard — WER@10, slope, best WER for all completed runs
python results/leaderboard.py

# Plot WER curves for a track
python results/plot_tracks.py --track B
python results/plot_tracks.py --track D
```

---

## Quick reference — local-only vs cloud-required

| Track | Local (RTX 4050 6 GB) | Cloud only |
|---|---|---|
| B | B0, B1, B2, B3, B4, B5 | — |
| C | C3 (Whisper+Qwen) | C1 (7B), C2 (5.6B) |
| D | D1b, D1d, D2a, D2d, D3b, D3c, D4 | — |
| E | E1a, E1b, E2b, E3 | — |