#!/usr/bin/env bash
# autoresearch/sweep2.sh — corrected sweep runner
# Uses clean PID capture (no stdout pollution).
# Skips experiments that already have a leaderboard row.
# Run: nohup bash autoresearch/sweep2.sh > /home/sweep2.log 2>&1 &

set -uo pipefail

REPO=/home/brain2text-experiments
TRAIN=data/toy_train.hdf5
VAL=data/toy_val.hdf5
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
export PYTHONIOENCODING=utf-8

cd "$REPO"

# ── helpers ────────────────────────────────────────────────────────────────

notify() {
    python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true
}

has_result() {
    local id="$1"
    python -c "
import sqlite3, sys
try:
    conn = sqlite3.connect('results/leaderboard.sqlite')
    rows = conn.execute(\"SELECT COUNT(*) FROM runs WHERE expt_id=? AND profile='toy'\", ('$id',)).fetchone()
    conn.close()
    sys.exit(0 if rows and rows[0] > 0 else 1)
except: sys.exit(1)
" 2>/dev/null
}

run_fg() {
    local id="$1"; shift
    local logf="/home/logs/${id}.log"
    if has_result "$id"; then
        echo "[sweep2] SKIP $id — already has leaderboard row" >&2
        return 0
    fi
    echo "[sweep2] START $id" >&2
    python run.py --expt "$id" --profile toy --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        > "$logf" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        notify "$id done — $(python -c "
import sqlite3
conn=sqlite3.connect('results/leaderboard.sqlite')
r=conn.execute(\"SELECT wer_at_ep10,slope FROM runs WHERE expt_id='$id' ORDER BY rowid DESC LIMIT 1\").fetchone()
conn.close()
print(f'WER@10={r[0]:.4f} slope={r[1]:.4f}') if r else print('no row')
" 2>/dev/null || echo '?')" ok
    else
        notify "$id FAILED (exit $rc)" fail
    fi >&2
    return $rc
}

# Background launcher — ONLY prints PID to stdout, all other output to stderr
bg() {
    local id="$1"; shift
    local logf="/home/logs/${id}.log"
    if has_result "$id"; then
        echo "[sweep2] SKIP $id — already has leaderboard row" >&2
        echo "0"  # fake PID that wait 0 will succeed on
        return 0
    fi
    echo "[sweep2] BG $id" >&2
    python run.py --expt "$id" --profile toy --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        > "$logf" 2>&1 &
    echo "$!"
}

parallel2() {
    local a="$1" b="$2"
    local pa pb
    pa=$(bg "$a")
    pb=$(bg "$b")
    [ "$pa" != "0" ] && wait "$pa" || true
    [ "$pb" != "0" ] && wait "$pb" || true
}

parallel3() {
    local a="$1" b="$2" c="$3"
    local pa pb pc
    pa=$(bg "$a")
    pb=$(bg "$b")
    pc=$(bg "$c")
    [ "$pa" != "0" ] && wait "$pa" || true
    [ "$pb" != "0" ] && wait "$pb" || true
    [ "$pc" != "0" ] && wait "$pc" || true
}

mkdir -p /home/logs
notify "sweep2.sh starting — Tracks D/E/C/F remaining (B0-B5 already running)" start

# ── Wait for B0_baseline to complete (needed for A1) ──────────────────────

echo "[sweep2] Waiting for B0_baseline to finish..." >&2
until has_result "B0_baseline"; do sleep 30; done
echo "[sweep2] B0_baseline done." >&2
python results/leaderboard.py --list 2>/dev/null || true

# ── Track A3 (phoneme probe — routing fixed) ──────────────────────────────
echo "[sweep2] === A3 phoneme probe ===" >&2
run_fg A3 || true

# ── Track D (loss ablations; all independent) ─────────────────────────────
echo "[sweep2] === Track D ===" >&2
parallel2 D1b D1d
parallel2 D2a D2d
parallel2 D3b D3c
run_fg D4 || true
notify "Track D complete" ok >&2
python results/leaderboard.py --list 2>/dev/null || true

# ── Track E (projectors) ──────────────────────────────────────────────────
echo "[sweep2] === Track E ===" >&2
parallel2 E1a E1b
run_fg E2b || true
run_fg E3 || true
notify "Track E complete" ok >&2
python results/leaderboard.py --list 2>/dev/null || true

# ── Wait for all B-track to complete, then A1 ─────────────────────────────
echo "[sweep2] Waiting for B-track to finish for A1..." >&2
for id in B1 B2 B3 B4 B5; do
    until has_result "$id" || ! pgrep -f "run.py.*--expt $id" > /dev/null 2>&1; do
        sleep 20
    done
done

echo "[sweep2] === A1 CKA (needs B0 checkpoint) ===" >&2
run_fg A1 || true
echo "[sweep2] A2 DEFERRED — corpora not present" >&2
notify "A2 deferred — BCI_DATA_ROOT / corpora not provided" warn

notify "Track A complete (A1 A3; A2 deferred; A4 derived from C rows)" ok >&2

# ── Track C (cloud decoders: C1/C2/C3 in parallel) ───────────────────────
echo "[sweep2] === Track C ===" >&2
parallel3 C1 C2 C3
notify "Track C complete" ok >&2
python results/leaderboard.py --list 2>/dev/null || true

# ── Track F: JEPA pretraining ─────────────────────────────────────────────
echo "[sweep2] === Track F: JEPA ===" >&2
run_fg F1 || true
parallel2 F2 F3

# Downstream fine-tunes
for fid in F1 F2 F3; do
    ckpt=$(find results/runs -name "pretrained_encoder.pth" -path "*${fid}*" 2>/dev/null | head -1 || true)
    if [ -n "$ckpt" ]; then
        echo "[sweep2] F downstream: $fid with $ckpt" >&2
        run_fg "$fid" "--override" "encoder.pretrained_ckpt=$ckpt" || true
    else
        echo "[sweep2] $fid downstream: no pretrained_encoder.pth found" >&2
        notify "$fid downstream skipped — no pretrained_encoder.pth" warn
    fi
done
notify "Track F complete" ok >&2
python results/leaderboard.py --list 2>/dev/null || true

# ── B2 retry (or defer) ────────────────────────────────────────────────────
# HRM DEQ dtype mismatch — DEFERRED (requires code fix for bf16 compat)
echo "[sweep2] B2 HRM DEFERRED: DEQ backward BFloat16/Float32 dtype mismatch" >&2
notify "B2 HRM DEFERRED: DEQ backward incompatible with bf16 autocast. Needs code fix." warn

# ── Full leaderboard ──────────────────────────────────────────────────────
echo "[sweep2] === FULL LEADERBOARD ===" >&2
python results/leaderboard.py --list 2>/dev/null || true

# ── Combination phase ─────────────────────────────────────────────────────
echo "[sweep2] === COMBINATION PHASE ===" >&2
# CMB-1: Best-B encoder + qformer(32) + D1b anneal
run_fg B1 "--override" "projector.variant=qformer projector.n_queries=32 loss.ctc_anneal.anneal_epochs=75 loss.contrastive.weight=0.0" || true
# CMB-2: Best-B encoder + qformer(32) + label smoothing
run_fg B1 "--override" "projector.variant=qformer projector.n_queries=32 loss.label_smoothing=0.1" || true
notify "Combination phase done — sweep COMPLETE" ok >&2

echo "[sweep2] === SWEEP COMPLETE at $(date) ===" >&2
