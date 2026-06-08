#!/usr/bin/env bash
# autoresearch/run_remaining.sh
# Runs D2a through F3 + downstream + combos, strictly sequentially in pairs.
# Assumes D1b and D1d are already running. Waits for them, then continues.
# Run: nohup bash autoresearch/run_remaining.sh > /home/remaining.log 2>&1 &

set -euo pipefail

REPO=/home/brain2text-experiments
TRAIN="$REPO/data/toy_train.hdf5"
VAL="$REPO/data/toy_val.hdf5"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
export PYTHONIOENCODING=utf-8

cd "$REPO"

slack() { python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true; }

lb() {
    local id="$1"
    python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
r = c.execute(\"SELECT wer_at_ep10, slope FROM runs WHERE expt_id='${id}' ORDER BY rowid DESC LIMIT 1\").fetchone()
c.close()
print(f'WER@10={r[0]:.4f} slope={r[1]:.4f}') if r and r[0] is not None else print('no-row')
" 2>/dev/null || echo "?"
}

expt() {
    local id="$1"; shift
    echo "[run] START $id $(date -u +%H:%M:%S)"
    python run.py --expt "$id" --profile toy --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        >> "/home/logs/${id}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[run] DONE $id — $(lb $id)"
        slack "$id done — $(lb $id)" ok
    else
        echo "[run] FAIL $id exit=$rc"
        slack "$id FAILED exit=$rc — see /home/logs/${id}.log" fail
    fi
    return $rc
}

expt_bg() {
    local id="$1"; shift
    echo "[run] BG $id $(date -u +%H:%M:%S)" >&2
    python run.py --expt "$id" --profile toy --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        >> "/home/logs/${id}.log" 2>&1 &
    echo "$!"
}

echo "[run] run_remaining.sh starting at $(date -u)"

# ── Wait for D1b and D1d to finish ──────────────────────────────────────────
echo "[run] Waiting for D1b and D1d to finish..."
while pgrep -f "run.py.*--expt D1" > /dev/null 2>&1; do sleep 15; done
echo "[run] D1b + D1d done."

# ── D2a + D2d ────────────────────────────────────────────────────────────────
echo "[run] === D2a + D2d ==="
p1=$(expt_bg D2a); p2=$(expt_bg D2d)
wait "$p1" || true; wait "$p2" || true

# ── D3b + D3c ────────────────────────────────────────────────────────────────
echo "[run] === D3b + D3c ==="
p1=$(expt_bg D3b); p2=$(expt_bg D3c)
wait "$p1" || true; wait "$p2" || true

# ── D4 ───────────────────────────────────────────────────────────────────────
echo "[run] === D4 ==="
expt D4 || true

slack "Track D complete — D1b D1d D2a D2d D3b D3c D4 done" ok
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'D%' AND profile='toy' AND wer_at_ep10 IS NOT NULL ORDER BY slope DESC\").fetchall()
c.close()
print('=== Track D ===')
for r in rows: print(f'  {r[0]}: WER@10={r[1]:.4f} slope={r[2]:.4f}')
" 2>/dev/null || true

# ── Track E ──────────────────────────────────────────────────────────────────
echo "[run] === E1a + E1b ==="
p1=$(expt_bg E1a); p2=$(expt_bg E1b)
wait "$p1" || true; wait "$p2" || true

echo "[run] === E2b ==="
expt E2b || true

echo "[run] === E3 ==="
expt E3 || true

slack "Track E complete — E1a E1b E2b E3 done" ok
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'E%' AND profile='toy' AND wer_at_ep10 IS NOT NULL ORDER BY slope DESC\").fetchall()
c.close()
print('=== Track E ===')
for r in rows: print(f'  {r[0]}: WER@10={r[1]:.4f} slope={r[2]:.4f}')
" 2>/dev/null || true

# ── Wait for B4 MoE to finish before A1 CKA (A1 needs GPU headroom) ─────────
echo "[run] Waiting for B4 MoE to finish before A1..."
while pgrep -f "run.py.*--expt B4" > /dev/null 2>&1; do sleep 15; done
echo "[run] B4 done. Starting A1."

# ── A1 CKA analysis ──────────────────────────────────────────────────────────
echo "[run] === A1 CKA ==="
expt A1 || true
echo "[run] A2 DEFERRED — corpora not present"
echo "[run] A3 DEFERRED — toy HDF5 has no phoneme labels"
echo "[run] A4 — derived from C1/C2 leaderboard rows (no separate run needed)"
slack "Track A: A1 done. A2/A3 deferred. A4 derived from C rows." ok

# ── Track C decoders ─────────────────────────────────────────────────────────
echo "[run] === C1 + C2 + C3 ==="
# C1 and C2 are 7B models — run in parallel (both fit 40GB)
p1=$(expt_bg C1); p2=$(expt_bg C2)
wait "$p1" || true; wait "$p2" || true
expt C3 || true

slack "Track C complete — C1 C2 C3 done" ok
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE 'C%' AND profile='toy' AND wer_at_ep10 IS NOT NULL ORDER BY slope DESC\").fetchall()
c.close()
print('=== Track C ===')
for r in rows: print(f'  {r[0]}: WER@10={r[1]:.4f} slope={r[2]:.4f}')
" 2>/dev/null || true

# ── Track F JEPA ──────────────────────────────────────────────────────────────
echo "[run] === F1 JEPA audio pretrain ==="
expt F1 || true

echo "[run] === F2 + F3 JEPA (parallel) ==="
p1=$(expt_bg F2); p2=$(expt_bg F3)
wait "$p1" || true; wait "$p2" || true

# Downstream fine-tunes
for fid in F1 F2 F3; do
    ckpt=$(find results/runs -name "pretrained_encoder.pth" -path "*${fid}*" 2>/dev/null | head -1 || true)
    if [ -n "$ckpt" ]; then
        echo "[run] $fid downstream fine-tune: $ckpt"
        expt "$fid" --override "encoder.pretrained_ckpt=$ckpt" || true
    else
        echo "[run] $fid downstream SKIP — pretrained_encoder.pth not found"
        slack "$fid downstream skipped — no pretrained_encoder.pth" warn
    fi
done

slack "Track F complete — F1 F2 F3 pretrain + downstream done" ok

# ── Full leaderboard ──────────────────────────────────────────────────────────
echo "[run] === FULL LEADERBOARD ==="
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id,wer_at_ep10,slope,best_wer FROM runs WHERE profile='toy' AND wer_at_ep10 IS NOT NULL ORDER BY expt_id, rowid DESC\").fetchall()
c.close()
print(f'{'Expt':<18} {'WER@10':>8} {'slope':>8} {'best_wer':>9}')
print('-' * 47)
seen = set()
for r in rows:
    if r[0] not in seen:
        print(f'{r[0]:<18} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>9.4f}')
        seen.add(r[0])
" 2>/dev/null || true

# ── Combination phase ──────────────────────────────────────────────────────────
echo "[run] === COMBINATION PHASE ==="
# CMB-1: B1 + qformer(32) + ctc_anneal + no contrastive
expt B1 \
    --override projector.variant=qformer \
    --override projector.n_queries=32 \
    --override "loss.ctc_anneal.anneal_epochs=75" \
    --override "loss.contrastive.weight=0.0" || true

# CMB-2: B1 + qformer(32) + label smoothing 0.1
expt B1 \
    --override projector.variant=qformer \
    --override projector.n_queries=32 \
    --override "loss.label_smoothing=0.1" || true

slack "SWEEP COMPLETE on A100 422377 — all tracks done. Check results/leaderboard.sqlite" ok
echo "[run] === SWEEP COMPLETE at $(date -u) ==="
