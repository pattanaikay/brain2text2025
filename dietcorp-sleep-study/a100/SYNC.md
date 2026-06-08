# A100 setup & data sync

Everything runs on a **fresh A100**. Local machine is only for code dev + CPU unit tests.

## 1. Create the instance
Use the `jarvislabs` skill / `jl` CLI (same workflow used by `train_ctc.py`'s auto-pause):
```bash
jl create --gpu A100 --name dietcorp-sleep     # or the dashboard
jl ssh dietcorp-sleep
```

## 2. Get the code there (standalone — no sibling repos needed)
```bash
# from local machine
rsync -avz dietcorp-sleep-study/  user@<instance>:/workspace/dietcorp-sleep-study/
# (or: git push a branch and clone on the box)
```

## 3. Get the data + checkpoint there
This study needs per-trial **neural + `seq_class_ids` + session** (the format `train_ctc.py` used).

- **Checkpoint** (88 MB): upload `best_model_per.pth` → `data/best_model_per.pth`.
- **Session stats**: upload `session_stats.json` → `data/session_stats.json`.
- **Data** (two options):
  - **A. Reuse formatted data** you already trained on (per-session `data_train.hdf5`/
    `data_val.hdf5` with `seq_class_ids`) → put under `data/sessions/`. This is the path that
    enables **all** conditions C0–C4.
  - **B. From Dryad**: download `doi:10.5061/dryad.x69p8czpq`, then format with DietCorp's
    `notebooks/formatCompetitionData.ipynb` to produce per-session files with `seq_class_ids`.

> Note: the local `preprocessed_data.h5` (17 GB) has only `neural` + `transcription` (no
> `seq_class_ids`). It supports C0/C1 (self-reference PER) but **not** C2/C4 directly. Use a
> formatted set for the full grid.

## 4. Environment
```bash
cd /workspace/dietcorp-sleep-study
bash a100/env_setup.sh
```

## 5. Verify + build LM + run
```bash
source .venv/bin/activate
python a100/prepare_data.py --data data/sessions          # confirms seq_class_ids present
python a100/build_lm.py     --data data/sessions --out_dir data/lm   # phoneme LM + lexicon
# edit configs/study.yaml paths if needed, then:
bash a100/run_matrix.sh
# run_matrix runs a phoneme-inventory self-check first: it decodes the GROUND-TRUTH
# seq_class_ids -> words and prints WER. If that WER is not ~0, core/phonemes.py does NOT
# match your data's formatting -> fix PHONEMES (or use DietCorp's own LM decoder) before
# trusting the C0-C4 WER numbers. PER numbers are unaffected.
```

## 6. Collect
- `results/study_results.json` — PER[condition][N][day] + latencies
- `results/wer_vs_day.png` — the headline figure
- Pull back: `rsync -avz user@<instance>:/workspace/dietcorp-sleep-study/results/ ./results/`

## 7. Pause/stop the instance (cost)
```bash
jl pause <instance_id> --yes        # train_ctc.py uses this exact call
```

## First-pass tip
Start small to validate end-to-end before the full grid:
```bash
python run_study.py --config configs/study.yaml \
  --conditions C0 C1 C2 --n_steps 0 1 2 --max_sessions 5 --max_trials 32
```
