#!/usr/bin/env bash
# autoresearch/sweep.sh
# Master sweep runner — Tracks A/B/C/D/E/F, toy-only, parallel batches.
# Run on A100 via: nohup bash autoresearch/sweep.sh > /home/sweep.log 2>&1 &
#
# All experiments use --profile toy.  B3_mamba SKIPPED (CUDA toolkit mismatch).
# Parallelism: 2-3 runs at once (40 GB / ~6 GB each).

set -euo pipefail

REPO=/home/brain2text-experiments
TRAIN=data/toy_train.hdf5
VAL=data/toy_val.hdf5
# SLACK_WEBHOOK_URL should be set via environment variable
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export PYTHONIOENCODING=utf-8

cd "$REPO"

# ── helpers ─────────────────────────────────────────────────────────────────

notify() { python autoresearch/notify_slack.py "$1" --status "${2:-ok}" || true; }

run_expt() {
    local id="$1"
    local extra="${2:-}"
    local logf="/home/logs/${id}.log"
    echo "[sweep] Starting $id at $(date)"
    python run.py --expt "$id" --profile toy \
        --train_h5 "$TRAIN" --val_h5 "$VAL" $extra \
        > "$logf" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        local row
        row=$(python -c "
import sqlite3, json
conn = sqlite3.connect('results/leaderboard.sqlite')
rows = conn.execute(\"SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs WHERE expt_id=? ORDER BY rowid DESC LIMIT 1\", ('$id',)).fetchall()
conn.close()
if rows:
    r = rows[0]
    print(f'WER@10={r[1]:.4f} slope={r[2]:.4f} best_wer={r[3]:.4f}')
else:
    print('no leaderboard row')
" 2>/dev/null || echo "leaderboard unavailable")
        notify "$id done — $row" ok
        echo "[sweep] $id DONE — $row"
    else
        notify "$id FAILED (exit $rc) — see /home/logs/${id}.log" fail
        echo "[sweep] $id FAILED exit=$rc"
    fi
    return $rc
}

run_expt_bg() {
    local id="$1"
    local extra="${2:-}"
    local logf="/home/logs/${id}.log"
    echo "[sweep] Launching $id in background"
    python run.py --expt "$id" --profile toy \
        --train_h5 "$TRAIN" --val_h5 "$VAL" $extra \
        > "$logf" 2>&1 &
    echo $!
}

wait_bg() {
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    return $failed
}

# ── Setup ────────────────────────────────────────────────────────────────────

mkdir -p /home/logs
notify "Phase 2 sweep starting — Tracks A/B/C/D/E/F, toy, parallel" start

# ── BATCH 1: B0_baseline + A3 (parallel, B0 is control; A3 self-contained) ──

echo "[sweep] === BATCH 1: B0_baseline + A3 (parallel) ==="
p0=$(run_expt_bg B0_baseline)
p1=$(run_expt_bg A3)
wait_bg "$p0" "$p1" || true
notify "BATCH 1 done — B0_baseline + A3" ok

# Report leaderboard after B0
echo "[sweep] Leaderboard after Batch 1:"
python results/leaderboard.py --list 2>/dev/null || true

# ── BATCH 2: B1 + B2 (parallel; both depend on B0) ──────────────────────────

echo "[sweep] === BATCH 2: B1 + B2 (parallel) ==="
p0=$(run_expt_bg B1)
p1=$(run_expt_bg B2)
wait_bg "$p0" "$p1" || true
notify "BATCH 2 done — B1 + B2" ok

# ── BATCH 3: B3 + B4 (parallel; both depend on B0) ──────────────────────────

echo "[sweep] === BATCH 3: B3 + B4 (parallel) ==="
p0=$(run_expt_bg B3)
p1=$(run_expt_bg B4)
wait_bg "$p0" "$p1" || true
notify "BATCH 3 done — B3 + B4" ok

# ── BATCH 4: B5 ──────────────────────────────────────────────────────────────

echo "[sweep] === BATCH 4: B5 ==="
run_expt B5 || true

# ── Track B complete ─────────────────────────────────────────────────────────

notify "Track B complete — B0/B1/B2/B3/B4/B5 done (B3_mamba SKIPPED — CUDA mismatch)" ok
echo "[sweep] === Track B leaderboard ==="
python results/leaderboard.py --list 2>/dev/null || true

# ── BATCH 5: D1b + D1d (parallel; independent) ──────────────────────────────

echo "[sweep] === BATCH 5: D1b + D1d (parallel) ==="
p0=$(run_expt_bg D1b)
p1=$(run_expt_bg D1d)
wait_bg "$p0" "$p1" || true

# ── BATCH 6: D2a + D2d (parallel) ───────────────────────────────────────────

echo "[sweep] === BATCH 6: D2a + D2d (parallel) ==="
p0=$(run_expt_bg D2a)
p1=$(run_expt_bg D2d)
wait_bg "$p0" "$p1" || true

# ── BATCH 7: D3b + D3c (parallel) ───────────────────────────────────────────

echo "[sweep] === BATCH 7: D3b + D3c (parallel) ==="
p0=$(run_expt_bg D3b)
p1=$(run_expt_bg D3c)
wait_bg "$p0" "$p1" || true

# ── BATCH 8: D4 ──────────────────────────────────────────────────────────────

echo "[sweep] === BATCH 8: D4 ==="
run_expt D4 || true

# ── Track D complete ─────────────────────────────────────────────────────────

notify "Track D complete — D1b/D1d/D2a/D2d/D3b/D3c/D4 done" ok
echo "[sweep] === Track D leaderboard ==="
python results/leaderboard.py --list 2>/dev/null || true

# ── BATCH 9: E1a + E1b (parallel) ────────────────────────────────────────────

echo "[sweep] === BATCH 9: E1a + E1b (parallel) ==="
p0=$(run_expt_bg E1a)
p1=$(run_expt_bg E1b)
wait_bg "$p0" "$p1" || true

# ── BATCH 10: E2b ────────────────────────────────────────────────────────────

echo "[sweep] === BATCH 10: E2b ==="
run_expt E2b || true

# ── BATCH 11: E3 (depends on E2b) ────────────────────────────────────────────

echo "[sweep] === BATCH 11: E3 ==="
run_expt E3 || true

# ── Track E complete ─────────────────────────────────────────────────────────

notify "Track E complete — E1a/E1b/E2b/E3 done" ok
echo "[sweep] === Track E leaderboard ==="
python results/leaderboard.py --list 2>/dev/null || true

# ── BATCH 12: A1 (needs B0 checkpoint) ───────────────────────────────────────

echo "[sweep] === BATCH 12: A1 ==="
run_expt A1 || true

# A2: skip if corpora not present (check for BCI_DATA_ROOT / transcript files)
if [ -n "${BCI_DATA_ROOT:-}" ] && [ -f "${BCI_DATA_ROOT}/transcripts_val.txt" ]; then
    echo "[sweep] Running A2 — corpora found at $BCI_DATA_ROOT"
    run_expt A2 || true
else
    echo "[sweep] A2 DEFERRED — corpora not provided (BCI_DATA_ROOT not set or missing transcripts_val.txt)"
    notify "A2 deferred — corpora not provided" warn
fi

notify "Track A complete — A1 A3 done (A2 deferred; A4 derived from C rows)" ok

# ── BATCH 13: Track C decoders (parallel C1 + C2 + C3) ──────────────────────

echo "[sweep] === BATCH 13: C1 + C2 + C3 (parallel) ==="
p0=$(run_expt_bg C1)
p1=$(run_expt_bg C2)
p2=$(run_expt_bg C3)
wait_bg "$p0" "$p1" "$p2" || true

notify "Track C complete — C1/C2/C3 done" ok
echo "[sweep] === Track C leaderboard ==="
python results/leaderboard.py --list 2>/dev/null || true

# ── BATCH 14: Track F — JEPA pretraining ─────────────────────────────────────

echo "[sweep] === BATCH 14: F1 (audio JEPA pretrain) ==="
run_expt F1 || true

echo "[sweep] === BATCH 15: F2 + F3 (parallel; depend on F1) ==="
p0=$(run_expt_bg F2)
p1=$(run_expt_bg F3)
wait_bg "$p0" "$p1" || true

# Downstream fine-tunes for F1/F2/F3
echo "[sweep] === BATCH 16: F downstream fine-tunes ==="
for fid in F1 F2 F3; do
    ckpt=$(ls results/runs/${fid}_*/pretrained_encoder.pth 2>/dev/null | head -1 || true)
    if [ -n "$ckpt" ]; then
        echo "[sweep] Fine-tuning $fid backbone: $ckpt"
        run_expt "$fid" "--override encoder.pretrained_ckpt=$ckpt" || true
    else
        echo "[sweep] WARNING: No pretrained_encoder.pth found for $fid — skipping downstream"
        notify "$fid downstream skipped — pretrained_encoder.pth not found" warn
    fi
done

notify "Track F complete — F1/F2/F3 pretrain + downstream fine-tunes done" ok

# ── Final leaderboard ────────────────────────────────────────────────────────

echo "[sweep] === FULL LEADERBOARD ==="
python results/leaderboard.py --list 2>/dev/null || true

notify "All tracks A/B/C/D/E/F toy sweep COMPLETE — entering combination phase" ok

# ── COMBINATION PHASE ────────────────────────────────────────────────────────

echo "[sweep] === COMBINATION PHASE ==="
# Pick winners from leaderboard — read top-slope encoder, projector, loss
# CMB-1: Best-B encoder + E2b Q-Former + D1b anneal (conservative best guess)
# Will be updated by whoever reads the leaderboard after each track
# For now default to B1 (ConformerXL — strong hypothesis) + qformer + D1b

echo "[sweep] CMB-1: B1 + qformer(32) + D1b anneal"
run_expt B1 "--override projector.variant=qformer projector.n_queries=32 loss.ctc_anneal.anneal_epochs=75 loss.contrastive.weight=0.0" || true

echo "[sweep] CMB-2: Best-B + D4 label-smooth + qformer"
run_expt B1 "--override projector.variant=qformer projector.n_queries=32 loss.label_smoothing=0.1" || true

notify "Combination phase done — sweep COMPLETE on A100 422377" ok

echo "[sweep] === SWEEP COMPLETE at $(date) ==="
