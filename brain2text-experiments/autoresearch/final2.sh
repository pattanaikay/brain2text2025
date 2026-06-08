#!/usr/bin/env bash
# autoresearch/final2.sh — second autonomous heal pass + chart generation + pause
# Fixes applied in run.py before this script runs:
#   (1) enc.to(compute_dtype)  → fixes B2 HRM DEQ bf16/fp32 mismatch
#   (2) dec bridge .to(device) → fixes C3 Whisper bridge CPU/CUDA mismatch
# This script installs remaining packages, runs fixed experiments, generates
# charts, then pauses the instance. No human input required.

set -uo pipefail

REPO=/home/brain2text-experiments
TRAIN="$REPO/data/toy_train.hdf5"
VAL="$REPO/data/toy_val.hdf5"
MACHINE_ID=422523
JL_TOKEN="7H1cs4uhE6xRrzz7XsIHbodxFMleyd3h3wlHXE_A1oM"
JL_API="https://backendn.jarvislabs.net"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/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"
export PYTHONIOENCODING=utf-8

cd "$REPO"

slack() { python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true; }

lb() {
    local id="$1"
    python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
r = c.execute(\"SELECT wer_at_ep10,slope FROM runs WHERE expt_id='${id}' ORDER BY rowid DESC LIMIT 1\").fetchone()
c.close()
print(f'WER@10={r[0]:.4f} slope={r[1]:.4f}') if r and r[0] is not None else print('no-row')
" 2>/dev/null || echo "?"
}

run_expt() {
    local id="$1"; shift
    echo "[final2] START $id $(date -u +%H:%M:%S)"
    python run.py --expt "$id" --profile toy \
        --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        >> "/home/logs/${id}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[final2] DONE $id — $(lb $id)"
        slack "$id DONE — $(lb $id)" ok
    else
        tail -5 "/home/logs/${id}.log" >> /home/final2.log 2>/dev/null || true
        echo "[final2] FAIL $id exit=$rc"
        slack "$id FAILED exit=$rc" fail
    fi
    return $rc
}

pause_instance() {
    echo "[final2] Pausing $MACHINE_ID..."
    for attempt in 1 2 3; do
        resp=$(curl -s -X POST \
            -H "Authorization: Bearer $JL_TOKEN" \
            "$JL_API/misc/pause?machine_id=$MACHINE_ID" \
            --max-time 30 2>&1 || echo "err")
        echo "[final2] Pause attempt $attempt: $resp"
        sleep 5
        state=$(curl -s -H "Authorization: Bearer $JL_TOKEN" \
            "$JL_API/users/fetch/$MACHINE_ID" --max-time 10 2>/dev/null \
            | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('instance',{}).get('status','?'))" 2>/dev/null || echo "?")
        echo "[final2] State: $state"
        if [ "$state" = "Pausing" ] || [ "$state" = "Paused" ]; then break; fi
        sleep 15
    done
    slack "Instance $MACHINE_ID PAUSED. final2.sh complete. Check results/figures/ and results/leaderboard.sqlite." pause
}

slack "final2.sh starting: B2 C2 C3 B3_mamba fixes + charts. Will pause when done." start

# ── 1. Install flash-attn for C2 Phi-4-MM ────────────────────────────────────
echo "[final2] Installing flash-attn for C2..."
pip install flash-attn --no-build-isolation -q >> /home/flash_attn_install.log 2>&1
if [ $? -eq 0 ]; then
    echo "[final2] flash-attn installed OK."
    slack "flash-attn installed — C2 retry ready." ok
else
    echo "[final2] flash-attn install failed — C2 will likely still fail."
    slack "flash-attn install FAILED — C2 may still fail." warn
fi

# ── 2. Set up CUDA 13.0 nvcc for B3_mamba ────────────────────────────────────
echo "[final2] Setting up CUDA 13.0 for mamba-ssm..."
# The apt-get from final.sh installed the packages but didn't update the symlink
# Try to find cuda-13.0 nvcc in standard locations
if [ -f /usr/local/cuda-13.0/bin/nvcc ]; then
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[final2] Found CUDA 13.0 at $CUDA_HOME"
elif dpkg -l | grep -q "cuda-nvcc-13-0"; then
    # Package installed but in different location
    NVCC13=$(find /usr -name "nvcc" 2>/dev/null | grep "13" | head -1 || true)
    if [ -n "$NVCC13" ]; then
        export CUDA_HOME=$(dirname $(dirname "$NVCC13"))
        export PATH="$(dirname $NVCC13):$PATH"
        echo "[final2] Found nvcc-13 at $NVCC13"
    fi
fi

# Retry apt install of just nvcc if not found
if ! nvcc --version 2>/dev/null | grep -q "13\.0"; then
    echo "[final2] CUDA 13.0 nvcc not found. Trying to install cuda-nvcc-13-0..."
    apt-get install -y --no-install-recommends cuda-nvcc-13-0 2>/dev/null || true
    if [ -f /usr/local/cuda-13.0/bin/nvcc ]; then
        export CUDA_HOME=/usr/local/cuda-13.0
        export PATH="$CUDA_HOME/bin:$PATH"
    fi
fi

echo "[final2] nvcc version: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"

# Retry mamba-ssm build
echo "[final2] Building mamba-ssm causal-conv1d..."
pip install mamba-ssm causal-conv1d --no-build-isolation -q > /tmp/mamba_build.log 2>&1
mamba_rc=$?
if [ $mamba_rc -eq 0 ]; then
    echo "[final2] mamba-ssm installed!"
    slack "mamba-ssm installed — running B3_mamba." ok
    MAMBA_OK=1
else
    echo "[final2] mamba-ssm still failing. Last 10 lines:"
    tail -10 /tmp/mamba_build.log
    slack "B3_mamba SKIPPED — mamba-ssm build still fails after CUDA 13.0 attempt." fail
    MAMBA_OK=0
fi

# ── 3. Run B2 HRM (enc.to(compute_dtype) fix) ────────────────────────────────
echo "[final2] === B2 HRM Dual-Timescale (dtype fix) ==="
run_expt B2 || true

# ── 4. Run C3 Whisper-Qwen (bridge .to(device) fix) ─────────────────────────
echo "[final2] === C3 Whisper-Qwen (device fix) ==="
run_expt C3 || true

# ── 5. Run C2 Phi-4-MM (flash-attn fix) ──────────────────────────────────────
echo "[final2] === C2 Phi-4-MM (flash-attn fix) ==="
run_expt C2 || true

# ── 6. Run B3_mamba if build succeeded ───────────────────────────────────────
if [ "${MAMBA_OK:-0}" = "1" ]; then
    echo "[final2] === B3_mamba (true Mamba SSM) ==="
    run_expt B3_mamba || true
fi

# ── 7. D3b retry (TopoLoss — confirm device fix is stable) ───────────────────
echo "[final2] === D3b TopoLoss λ=0.001 (confirm fix) ==="
run_expt D3b || true

# ── 8. Generate all charts ────────────────────────────────────────────────────
echo "[final2] === Generating charts ==="
python autoresearch/make_charts.py 2>&1 | tee -a /home/final2.log
slack "Charts generated — results/figures/ has SVG+PNG for tracks B C D E F + master scatter." ok

# ── 9. Final leaderboard ──────────────────────────────────────────────────────
echo "[final2] === FINAL LEADERBOARD ==="
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs WHERE profile='toy' ORDER BY slope ASC NULLS LAST\").fetchall()
c.close()
seen = {}
for r in rows:
    if r[0] not in seen and r[1] is not None: seen[r[0]] = r
print(f'Rank  Expt               WER@10     slope    Label')
print('-'*62)
rank = 0
for r in sorted(seen.values(), key=lambda x: x[2]):
    rank += 1
    sl = r[2]
    label = 'STRONG' if sl < -0.010 else ('PROMISING' if sl < -0.005 else ('WEAK' if sl < 0.005 else 'INERT'))
    print(f'{rank:>4}  {r[0]:<19} {r[1]:>8.4f}  {sl:>9.5f}  {label}')
" 2>/dev/null | tee -a /home/final2.log

# Summary Slack message
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
def best(prefix):
    r = c.execute(f\"SELECT expt_id,wer_at_ep10,slope FROM runs WHERE expt_id LIKE '{prefix}%' AND profile='toy' AND wer_at_ep10 IS NOT NULL ORDER BY slope LIMIT 1\").fetchone()
    return r
b,d,e,cf = best('B'),best('D'),best('E'),best('C')
c.close()
enc  = f'{b[0]} slope={b[2]:.4f}' if b else 'n/a'
loss = f'{d[0]} slope={d[2]:.4f}' if d else 'n/a'
proj = f'{e[0]} slope={e[2]:.4f}' if e else 'n/a'
dec  = f'{cf[0]} slope={cf[2]:.4f}' if cf else 'C-track failed'
print(f'FINAL: encoder={enc} | loss={loss} | projector={proj} | decoder={dec}. Charts ready.')
" 2>/dev/null | xargs -I{} python autoresearch/notify_slack.py {} --status ok 2>/dev/null || true

# ── 10. Pause ─────────────────────────────────────────────────────────────────
pause_instance
echo "[final2] COMPLETE at $(date -u)"
