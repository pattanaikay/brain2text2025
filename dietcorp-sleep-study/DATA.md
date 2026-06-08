# Data — what each condition needs and how to get it

## The two data fields
| Field | What | Where it is |
|-------|------|-------------|
| `neural` + `transcription` | raw signal + sentence text | **Local**: `brain2text-modeltraining/data/preprocessed_data.h5` (17 GB, 10,948 trials) |
| `seq_class_ids` | per-trial phoneme labels (41-class; 0..40) | **NOT local** — only dummy data has them; the real ones were on the training volume |

The trained head is **42 classes = blank(0) + 41 phonemes** (confirmed by `make_dummy_data.py`:
`PHONEME_VOCAB = 41`). The loader adds `+1`, so stored `seq_class_ids` are `0..40`.

## What each condition needs
| Condition | Needs `seq_class_ids`? | Runs on the local 17 GB file? |
|-----------|------------------------|-------------------------------|
| C0 no-adapt | No (self-reference PER) | ✅ yes |
| C1 self-label | No | ✅ yes |
| C2 LM-refined | **Yes** (to train the phoneme n-gram) | only with generated/obtained labels |
| C3 LM+memory+replay | **Yes** | only with generated/obtained labels |
| C4 oracle | **Yes** | only with generated/obtained labels |

## Getting `seq_class_ids` — two options

### Option A — generate locally from transcription (no Dryad)  ← try first
```bash
python a100/make_seq_class_ids.py --in data/preprocessed_data.h5 --out data/sessions_g2p.h5
# then VALIDATE the inventory against the trained head (decisive check):
python a100/make_seq_class_ids.py --in data/sessions_g2p.h5 \
    --validate_with_model data/best_model_per.pth --session_stats data/session_stats.json
```
- g2p_en(transcription) → ARPAbet → 41-class ids (the `core/phonemes.PHONEMES` ordering), SIL between words.
- **Validation:** the script decodes clean trials with the trained model and prints
  `PER(model_greedy vs g2p_label)`. **Low PER ⇒ the inventory/ordering match the head ⇒ C2/C4 trustworthy.**
  High PER ⇒ ordering is wrong ⇒ use Option B.
- **Caveat:** the head was trained on the *authors'* phonemization; g2p_en is an approximation,
  so this is an *approximate* oracle (good enough to test the mechanism; not the canonical number).

### Option B — re-obtain the authors' formatted data (faithful)
- **Restore** the per-session `data_train.hdf5`/`data_val.hdf5` (with `seq_class_ids`) from the
  training volume — `h5_list.json` shows they lived at `/home/data/.../hdf5_data_final/t15.*/`.
- **Or** download Dryad `doi:10.5061/dryad.x69p8czpq` and run DietCorp's
  `notebooks/formatCompetitionData.ipynb` to regenerate them. Guaranteed-correct labels.

## Recommendation
1. Run **C0/C1 now** on the 17 GB file to confirm the drift curve + the pipeline end-to-end.
2. Run **Option A** + validation. If PER validates → run C2/C3/C4 on `sessions_g2p.h5`.
3. For the **final, citable WER**, use **Option B** (authors' labels + ideally DietCorp's own
   n-gram word decoder) so the numbers are canonical.
