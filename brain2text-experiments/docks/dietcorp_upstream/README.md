# docks/dietcorp_upstream/ — vendored upstream (placeholder)

This directory is the **untouched** vendor slot for
[ebrahimfeghhi/transformers_with_dietcorp](https://github.com/ebrahimfeghhi/transformers_with_dietcorp).

It is intentionally empty in git. Vendor the real repo as a submodule:

```bash
git submodule add \
  https://github.com/ebrahimfeghhi/transformers_with_dietcorp \
  brain2text-experiments/docks/dietcorp_upstream
```

Then pin its commit SHA + content hash in [`../PINS.txt`](../PINS.txt):

```bash
python -c "from docks.dietcorp_dock import upstream_hash; print(upstream_hash())"
```

## Why a separate untouched slot?

The integration strategy (ADHD convergence, 2026-05-30) is **vendor-untouched
+ thin adapter dock**. All seam-cutting lives in
[`../dietcorp_dock.py`](../dietcorp_dock.py), never in the upstream files, so:

- the reproduction stays faithful (you run the original training loop), and
- upstream drift is detected by re-hashing, not hidden inside our edits.

## The two seams to cut (TODO markers in `dietcorp_dock.py`)

1. **Data shape** — map the repo's speech-BCI input tensor to/from our
   `(B, 240, 512)` neural windows. Default smoke uses the repo's OWN toy data
   so a shape mismatch on our side never invalidates the reproduction.
2. **Day/session index** — locate where the per-day affine recalibration index
   enters the upstream model. The reusable version is reimplemented natively in
   [`../../stages/projector/dietcorp_recal.py`](../../stages/projector/dietcorp_recal.py).

## Smoke

```bash
python -c "from docks.dietcorp_dock import run_smoke_iters; print(run_smoke_iters(2))"
```

Runs 2 iterations and prints losses. Once vendored, switch the smoke to drive
the upstream loop and diff against `tools/dietcorp_paper_oracle.yaml`.
