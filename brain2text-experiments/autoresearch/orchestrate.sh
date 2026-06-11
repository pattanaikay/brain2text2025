#!/usr/bin/env bash
# autoresearch/orchestrate.sh — simple sequential+parallel sweep orchestrator
# Runs remaining experiments (D/E/A/C/F + combinations) AFTER Track B finishes.
# Polls until Track B processes exit, then launches in batches of 2-3.
# Run: nohup bash autoresearch/orchestrate.sh > /home/orch.log 2>&1 &

REPO=/home/brain2text-experiments
TRAIN=data/toy_train.hdf5
VAL=data/toy_val.hdf5
# SLACK_WEBHOOK_URL should be set via environment variable
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export PYTHONIOENCODING=utf-8

cd "$REPO"

slack() { python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true; }

run() {
    local id="$1"; shift
    local logf="/home/logs/${id}.log"
    echo "[orch] START $id $(date -u +%H:%M:%S)"
    python run.py --expt "$id" --profile toy \
        --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        >> "$logf" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        row=$(python -c "import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite'); r=c.execute(\"SELECT wer_at_ep10,slope FROM runs WHERE expt_id='${id}' ORDER BY rowid DESC LIMIT 1\").fetchone(); c.close(); print(f'WER@10={r[0]:.4f} slope={r[1]:.4f}') if r else print('?')" 2>/dev/null || echo '?')
        echo "[orch] DONE $id — $row"
        slack "$id done — $row" ok
    else
        echo "[orch] FAIL $id (exit $rc)"
        slack "$id FAILED exit=$rc" fail
    fi
    return $rc
}

# ── Step 1: Wait for all Track B run.py processes to finish ───────────────
echo "[orch] Waiting for Track B (B0/B1/B3/B4/B5) to finish..."
while pgrep -f "run.py.*--expt B" > /dev/null 2>&1; do
    sleep 20
done
echo "[orch] All Track B done at $(date -u +%H:%M:%S)"
python -c "import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite'); [print(r) for r in c.execute(\"SELECT expt_id,profile,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'B%' ORDER BY rowid DESC LIMIT 10\").fetchall()]; c.close()" 2>/dev/null || true
slack "Track B complete. Starting D/E/A/C/F tracks." ok

# ── Step 2: A3 phoneme probe (fast, no GPU-heavy decoder) ─────────────────
echo "[orch] === A3 phoneme probe ==="
run A3 || true

# ── Step 3: Track D in parallel pairs ─────────────────────────────────────
echo "[orch] === Track D ==="
run D1b & p1=$!; run D1d & p2=$!; wait $p1; wait $p2
run D2a & p1=$!; run D2d & p2=$!; wait $p1; wait $p2
run D3b & p1=$!; run D3c & p2=$!; wait $p1; wait $p2
run D4 || true
slack "Track D complete" ok
python -c "import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite'); [print(r) for r in c.execute(\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'D%' ORDER BY slope DESC\").fetchall()]; c.close()" 2>/dev/null || true

# ── Step 4: Track E ────────────────────────────────────────────────────────
echo "[orch] === Track E ==="
run E1a & p1=$!; run E1b & p2=$!; wait $p1; wait $p2
run E2b || true
run E3 || true
slack "Track E complete" ok
python -c "import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite'); [print(r) for r in c.execute(\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'E%' ORDER BY slope DESC\").fetchall()]; c.close()" 2>/dev/null || true

# ── Step 5: A1 CKA (needs B0 checkpoint) ─────────────────────────────────
echo "[orch] === A1 CKA ==="
run A1 || true
echo "[orch] A2 DEFERRED — corpora not present"
slack "A2 deferred — BCI_DATA_ROOT corpora not provided" warn

# ── Step 6: Track C decoders ──────────────────────────────────────────────
echo "[orch] === Track C ==="
run C1 & p1=$!; run C2 & p2=$!; wait $p1; wait $p2
run C3 || true
slack "Track C complete" ok
python -c "import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite'); [print(r) for r in c.execute(\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'C%' ORDER BY slope DESC\").fetchall()]; c.close()" 2>/dev/null || true

# ── Step 7: Track F JEPA pretraining ──────────────────────────────────────
echo "[orch] === Track F ==="
run F1 || true
run F2 & p1=$!; run F3 & p2=$!; wait $p1; wait $p2

# Downstream fine-tunes for each JEPA backbone
for fid in F1 F2 F3; do
    ckpt=$(find results/runs -name "pretrained_encoder.pth" -path "*${fid}*" 2>/dev/null | head -1 || true)
    if [ -n "$ckpt" ]; then
        echo "[orch] $fid downstream with $ckpt"
        run "$fid" --override "encoder.pretrained_ckpt=$ckpt" || true
    else
        echo "[orch] $fid downstream SKIP — no pretrained_encoder.pth"
        slack "$fid downstream skipped — pretrained_encoder.pth not found" warn
    fi
done
slack "Track F complete" ok

# ── Step 8: B2 retry ──────────────────────────────────────────────────────
# HRM DEQ dtype mismatch — DEFERRED
echo "[orch] B2 HRM DEFERRED: DEQ bf16/fp32 mismatch. Marking as SKIP."
slack "B2 HRM DEFERRED: DEQ backward incompatible with bf16 autocast — code fix needed" warn

# ── Step 9: Full leaderboard ──────────────────────────────────────────────
echo "[orch] === FULL LEADERBOARD ==="
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute('SELECT expt_id,profile,wer_at_ep10,slope,best_wer FROM runs WHERE profile=\"toy\" ORDER BY slope DESC').fetchall()
c.close()
print(f'{'Expt':<16} {'WER@10':>8} {'slope':>8} {'best_wer':>9}')
print('-'*45)
for r in rows:
    print(f'{r[0]:<16} {str(r[2]):>8} {str(r[3]):>8} {str(r[4]):>9}')
" 2>/dev/null || true

# ── Step 10: Combination phase ────────────────────────────────────────────
echo "[orch] === COMBINATION PHASE ==="
# CMB-1: B1 encoder + qformer(32) + ctc_anneal + no contrastive
run B1 --override "projector.variant=qformer" \
    --override "projector.n_queries=32" \
    --override "loss.ctc_anneal.anneal_epochs=75" \
    --override "loss.contrastive.weight=0.0" || true

# CMB-2: B1 encoder + qformer(32) + label smoothing 0.1
run B1 --override "projector.variant=qformer" \
    --override "projector.n_queries=32" \
    --override "loss.label_smoothing=0.1" || true

slack "Sweep COMPLETE on A100 422377 — all tracks done" ok
echo "[orch] === SWEEP COMPLETE at $(date -u) ==="
