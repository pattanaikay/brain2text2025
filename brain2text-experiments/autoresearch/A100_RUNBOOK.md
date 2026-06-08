# A100 Runbook — Running the Sweep on JarvisLabs

> **Scope of the current sweep:** all of Tracks **A, B, C, D, E** run on the A100, all as
> **short `--profile toy` runs**. There are **no full / 150-epoch runs** in this sweep — the
> goal is to find the best building block per stage. The full-run tables in §2.3 / §2.4 are kept
> only as reference for the **separate** long-term-implementation effort, not this diagnostic sweep.

**When to use this:** complete track coverage (including `B3_mamba`, `A4`, `C1`/`C2`, `C3`)
without any Windows-specific bottlenecks — the fastest path to a ranked set of building blocks.

---

## 1. What changes on A100 vs local

| Bottleneck | Local (Windows, RTX 4050, 6 GB) | A100 (JarvisLabs, Linux, 40 GB) |
|---|---|---|
| `mamba-ssm` / `causal-conv1d` | Won't build on Windows | `pip install mamba-ssm causal-conv1d` → ~3 min |
| `bitsandbytes` 4-bit | Fragile on Windows | Works out of box on the PyTorch template |
| 7B+ models (A1 7B rows, A4, C1, C2) | OOM (6 GB) | Fits (40 GB) — all experiments runnable |
| `num_workers` | Must be 0 (spawn deadlock) | 8 (fork, fast dataloader) |
| `PYTHONIOENCODING` | Must be forced to utf-8 | UTF-8 by default on Linux |
| Quantize for LLM | Required (4-bit NF4, 6 GB) | Optional — `full.yaml` sets `quantize: false` |
| Parallel runs | One at a time (6 GB) | 2–3 simultaneous toy runs (40 GB) |
| `B3_mamba` | Deferred | Runnable after `pip install mamba-ssm` |
| `A4` audio-vs-vision E2E | Deferred | Runnable (7B model fits) |
| `C1` Qwen2-Audio-7B | Deferred | Runnable |
| `C2` Phi-4-MM | Deferred | Runnable |

**Net result:** all 28 A/B/C/D/E/F experiments become locally runnable on the A100.

---

## 2. Time estimates

### 2.1 Setup (one-time per session)

| Step | Time |
|---|---|
| Resume or create JarvisLabs instance | 2–5 min |
| SSH + git pull / jl upload | 1–2 min |
| `pip install -r requirements.txt` (PyTorch template has torch/cuda) | 3–5 min |
| `pip install mamba-ssm causal-conv1d` | 2–4 min |
| `py -3 autoresearch/preflight.py` (env probe + shape gate) | ~1 min |
| **Total setup** | **~10–15 min** |

### 2.2 Toy sweep per track

On A100 (40 GB, `--profile toy`): `batch_size=1`, `max_batches=200`, `epochs=20`.
The A100 is ~4–5× faster per epoch than the RTX 4050 at equivalent batch size,
so local toy estimates (from `registry.yaml`) compress accordingly.

| Track | Experiments | Est. time each | Track total |
|---|---|---|---|
| **A** (analysis) | A1 CKA, A2 PPL, A3 probe | 6 / 4 / 10 min | **~20 min** |
| **A4** (E2E, cloud-only locally) | A4 audio-vs-vision | ~45 min | **~45 min** |
| **B** (encoder sweep) | B0 + B1 + B2 + B3-gru + B3-mamba + B4 + B5 | ~5–8 min each | **~45 min** |
| **C** (decoder extras) | C1 + C2 + C3 | ~40 / ~35 / ~8 min | **~83 min** |
| **D** (loss ablations) | D1b, D1d, D2a, D2d, D3b, D3c, D4 | ~5 min each | **~35 min** |
| **E** (projector) | E1a, E1b, E2b + E3 grid (6 cells) | ~5 min + 6×5 min | **~45 min** |
| **F** (JEPA) | F1/F2/F3 pretrain (no LLM, fast) + 3 downstream fine-tunes | ~4 min + ~6 min each | **~30 min** |
| **Combination phase** | 3–5 composed runs | ~5–8 min each | **~30 min** |

**Total toy sweep: ~330 min ≈ 5.5 hours sequential** (incl. Track F pretrain + downstream).

Running 2–3 toy experiments in parallel (A100 has 40 GB; each toy run peaks at ~4–6 GB):
**~2.5–3 hours parallel.**

### 2.3 Full runs (after promotion)

Experiments that clear the ≥5% slope threshold in toy go to `--profile full`:

| Experiment | Full-run time (A100, 150 epochs) |
|---|---|
| Any Track B encoder | ~1.5–2 h |
| Track D / E combinations | ~1.5 h |
| A4 / C1 / C2 (7B decoders) | ~2.5–3 h |
| Combination candidate (best enc × proj × loss) | ~2 h |

If ~3 experiments promote: **~6 h of full runs.** Budget ~1 day total (overnight + morning).

### 2.4 Cost estimate

JarvisLabs A100 40 GB: **~$1.99–$2.50 / hr** (on-demand; spot ~$0.80–$1.00/hr).

| Scenario | Wall-clock | Cost |
|---|---|---|
| Toy sweep only, sequential | ~5 h | ~$10–12 |
| Toy sweep, parallel (2–3 runs) | ~2.5 h | ~$5–7 |
| Toy + 3 full runs (best candidates) | ~9–10 h | ~$18–25 |
| Complete overnight run (everything) | ~14–16 h | ~$28–40 |

**Recommendation (this sweep):** run the full A/B/C/D/E/F toy sweep in parallel (~2.5–3 h, **~$7–9**),
then pause. No full runs here — the deliverable is a ranked set of building blocks, so total cost
is just the toy-sweep figure. (Full-run costs above are reference for the separate long-term goal.)

---

## 3. Dependency resolution steps

### 3.1 Create / resume instance

```powershell
# Check current state
jl status --json

# Resume existing instance (instance from profiles/full.yaml)
jl resume 413754 --yes --json

# Or create a fresh A100
jl create --gpu A100 --storage 50 --yes --json
```

### 3.2 Upload the repo

```powershell
# From local machine:
jl upload <machine_id> "C:\Projects\Brain2Text2025\brain2text2025\brain2text-experiments" /home/brain2text-experiments

# Or use git (preferred for reproducibility):
# jl exec <id> -- sh -c "git clone <your-repo-url> /home/brain2text-experiments"
```

### 3.3 Install dependencies (one-shot)

```bash
# On the A100 instance (via jl exec or SSH):
cd /home/brain2text-experiments

# Base requirements (most already on JarvisLabs PyTorch template)
pip install -r requirements.txt

# Mamba (Linux + CUDA — installs cleanly here, never on Windows)
pip install mamba-ssm causal-conv1d

# Verify
python -c "import mamba_ssm; print('mamba ok')"
python -c "import bitsandbytes as bnb; print('bnb ok')"
```

Expected output: `mamba ok`, `bnb ok`. Total: **~5–8 min**.

### 3.4 Update full.yaml to match the actual instance

`profiles/full.yaml` has `instance_id: "413754"` — update if using a different instance:

```bash
sed -i 's/instance_id: "413754"/instance_id: "<new_id>"/' profiles/full.yaml
```

Or just leave it; `auto_pause` calls `jl pause $(jl status --json | jq .machine_id)` —
edit `results/leaderboard.py` if needed.

### 3.5 Run preflight to confirm all-clear

```bash
cd /home/brain2text-experiments
$env:PYTHONIOENCODING = "utf-8"      # still set it, harmless on Linux
python autoresearch/preflight.py
```

Expected: all experiments `RUNNABLE`, `bnb 4-bit: ok`, `mamba-ssm: available`.

---

## 4. Running the sweep on A100

### Option A — sequential (safest, easiest to monitor)

```bash
cd /home/brain2text-experiments

# Track B — baseline first (A1 below needs the B0 checkpoint), then encoders
python run.py --expt B0_baseline --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
for expt in B1 B2 B3 B3_mamba B4 B5; do
    python run.py --expt $expt --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
done

# Track A analysis (run.py routes A1/A2/A3 to their tools). A3 is self-contained; A1 needs the
# B0 checkpoint (run after B0); A2 needs corpora under $BCI_DATA_ROOT (skip if absent).
for expt in A3 A1 A4; do
    python run.py --expt $expt --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
done

# Track D — loss ablations (independent, any order)
for expt in D1b D1d D2a D2d D3b D3c D4; do
    python run.py --expt $expt --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
done

# Track E — projectors
python run.py --expt E1a --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
python run.py --expt E1b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
python run.py --expt E2b --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
python run.py --expt E3  --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5

# Track C (the cloud-only decoders)
python run.py --expt C1  --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
python run.py --expt C2  --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
python run.py --expt A4  --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5

# Track F — JEPA: pretrain each modality (F1 first; F2/F3 depend on it), then fine-tune each
# saved backbone downstream for the cross-modality WER comparison.
for expt in F1 F2 F3; do
    python run.py --expt $expt --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
done
# Each pretrain run prints its backbone path; fine-tune it via --override (real WER), e.g.:
#   python run.py --expt F1 --profile toy \
#     --override encoder.pretrained_ckpt=results/runs/<F1_dir>/pretrained_encoder.pth \
#     --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
```

### Option B — parallel (2 processes at once; ~40 GB / 2 ≈ 20 GB each with margin)

```bash
# B1 and B2 simultaneously (different encoder families, no GPU contention)
python run.py --expt B1 --profile toy ... &
python run.py --expt B2 --profile toy ... &
wait

# D-track — all 7 runs in 3 parallel batches
python run.py --expt D1b --profile toy ... &
python run.py --expt D1d --profile toy ... &
wait
python run.py --expt D2a --profile toy ... &
python run.py --expt D2d --profile toy ... &
wait
# etc.
```

### Option C — autoresearch sweep (recommended once sweep.py is built)

```bash
python autoresearch/sweep.py --profile toy \
    --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5 \
    --cloud  # lifts the local_ok filter for cloud-only experiments
```

---

## 5. After the sweep — reading results

```bash
# Leaderboard with WER@10, slope, best_wer, ranked by slope
python results/leaderboard.py --track B --sort slope

# Decide promotions (apply program.md promote/discard rules)
python results/leaderboard.py --promote-check --baseline B0_baseline

# Full-run the winners
python run.py --expt B1 --profile full \
    --train_h5 data/data_train.hdf5 --val_h5 data/val.hdf5
# → auto-pauses instance when done (profiles/full.yaml: auto_pause: true)
```

---

## 6. Pause / cost control

The `profiles/full.yaml` has `auto_pause: true` — `run.py` pauses the JarvisLabs instance
automatically when a `--profile full` run completes. For toy sweeps (which run sequentially
for hours), either:

```powershell
# Monitor from local machine
jl run logs <run_id> --tail 30

# Or add at the end of the sweep script:
jl pause 413754 --yes
```

Do not leave the instance running idle after the sweep. Paused instances bill storage only
(~$0.03/GB/day → ~$1.50/day for 50 GB).

---

## 7. Recommendation

**This sweep runs entirely on the A100, toy-only.** Setup ~15 min; the full A/B/C/D/E/F toy sweep
runs in ~2.5 h parallel (**~$6–8**); then pause. You wake up to a ranked leaderboard, Tufte charts,
and per-track interpretation reports identifying the best encoder / decoder / loss / projector.

No full runs are part of this diagnostic sweep. When you later pursue the long-term implementation,
the full-run tables in §2.3 / §2.4 estimate that separate effort — but that is a different goal,
driven by your decision after you see which building blocks won here.
