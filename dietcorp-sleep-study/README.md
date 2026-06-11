# dietcorp-sleep-study

Standalone study: **does deeper "sleep" memory consolidation in DietCorp's test-time
adaptation reduce WER under day-to-day electrode drift, at constant wake latency?**

This folder is **fully self-contained** — it copies the model + adaptation code it needs and
has **no import dependency** on `brain2text-experiments`. It is decoupled from the 25-experiment
sweep framework. Built to run **entirely on an A100** (local machine = code dev + CPU unit tests).

## What we take from the three papers
- **DietCorp** (arXiv:2507.02800): per-trial TTA — update only the patch-embedding from a CTC
  pseudo-label. We generalise its **single** AdamW step to **N** steps.
- **"Do Language Models Need Sleep"** (arXiv:2605.26099): offline N-pass consolidation at the
  eviction boundary; strict **wake/sleep separation** (wake = 1 forward pass); gains grow with N
  on *sequential* problems. (We use the principle + N-scaling; **not** their learned-rule arm —
  our rule is fixed gradient descent.)
- **ZenBrain** (arXiv:2604.23878): episodic memory + **Simulation-Selection** replay
  (priority = |surprise| + confidence + novelty).

## The experiment grid — WER[day, N, condition]
| ID | Condition | Proves |
|----|-----------|--------|
| C0 | no-adapt (N=0) | the drift baseline |
| C1 | self-label, N∈{1,2,4,8} | does collapse persist on real data? |
| C2 | LM-refined, N∈{1,2,4,8} | **H_main**: deeper N helps with good labels |
| C3 | LM + memory + replay, N∈{1,2,4,8} | the ZenBrain contribution |
| C4 | oracle labels, N∈{1,2,4,8} | the achievable ceiling |

H_main confirmed iff C2 (N>1) < C1 (N=1) at later days **and wake latency is flat across N**.

## Layout
```
core/        bit_encoder.py · model.py · episodic_memory.py · consolidator.py
             lm_refine.py · replay.py · drift_eval.py
configs/     study.yaml
run_study.py standalone driver (WER grid)
a100/        env_setup.sh · prepare_data.py · build_lm.py · run_matrix.sh
             collect_and_plot.py · SYNC.md
tests/       CPU unit tests
```

## Quick start
```bash
# Local CPU tests
py -3 -m pytest tests -q

# A100 (see a100/SYNC.md for instance create + data download first)
bash a100/env_setup.sh
python a100/download_data.py --out data/sessions          # Dryad dncjsxm85 (T15)
python a100/prepare_data.py --data data/sessions          # verify seq_class_ids present
python a100/build_lm.py     --data data/sessions --out_dir data/lm
bash a100/run_matrix.sh
python a100/collect_and_plot.py results/
```

## Data note
Phoneme labels (`seq_class_ids`) are **provided by the competition dataset** (no g2p). The
checkpoint `best_model_per.pth` = `CTCPhonemeModel.state_dict()` (`encoder.*` + `head.*`, 42
classes: 0=blank, 1-41=phonemes). Oracle labels and the phoneme n-gram LM both use the
dataset's `seq_class_ids` directly.
