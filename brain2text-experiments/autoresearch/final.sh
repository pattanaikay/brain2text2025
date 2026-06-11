#!/usr/bin/env bash
# autoresearch/final.sh — fully autonomous: heals, retries, pauses.
# No human approval needed. Runs after the main sweep completes.
# Launched via: nohup bash autoresearch/final.sh > /home/final.log 2>&1 &

set -uo pipefail

REPO=/home/brain2text-experiments
TRAIN="$REPO/data/toy_train.hdf5"
VAL="$REPO/data/toy_val.hdf5"
MACHINE_ID=422377
# JL_TOKEN and SLACK_WEBHOOK_URL should be set via environment variables
JL_TOKEN="${JL_TOKEN:-}"
JL_API="https://backendn.jarvislabs.net"
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export PYTHONIOENCODING=utf-8

cd "$REPO"

# ── helpers ──────────────────────────────────────────────────────────────────

slack() {
    python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true
}

lb_row() {
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
    echo "[final] START $id $(date -u +%H:%M:%S)"
    python run.py --expt "$id" --profile toy \
        --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        >> "/home/logs/${id}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[final] DONE $id — $(lb_row $id)"
        slack "$id DONE — $(lb_row $id)" ok
    else
        tail -5 "/home/logs/${id}.log" >> /home/final.log 2>/dev/null || true
        echo "[final] FAIL $id exit=$rc"
        slack "$id FAILED exit=$rc — skipping" fail
    fi
    return $rc
}

pause_instance() {
    echo "[final] Pausing instance $MACHINE_ID via JarvisLabs API..."
    resp=$(curl -s -X POST \
        -H "Authorization: Bearer $JL_TOKEN" \
        "$JL_API/misc/pause?machine_id=$MACHINE_ID" \
        --max-time 30 2>&1 || echo "curl_error")
    echo "[final] Pause API response: $resp"
    if echo "$resp" | grep -qi '"success"\|paused\|true'; then
        slack "Instance $MACHINE_ID PAUSED. Sweep complete. Check results/leaderboard.sqlite." pause
        echo "[final] Instance paused successfully."
    else
        slack "WARN: Auto-pause may have failed ($resp). Please pause manually: jl pause $MACHINE_ID --yes" warn
        echo "[final] WARNING: Pause response unclear — check manually."
    fi
}

# ── Step 0: Wait for all current sweep runs to finish ────────────────────────
echo "[final] Waiting for all current run.py processes to finish..."
while pgrep -f "run.py" > /dev/null 2>&1; do
    sleep 20
done
echo "[final] All prior runs finished at $(date -u)"

# Also wait for cleanup.sh and run_remaining.sh
while pgrep -f "cleanup.sh\|run_remaining.sh" > /dev/null 2>&1; do
    sleep 10
done
echo "[final] All orchestrator scripts finished at $(date -u)"

slack "Final auto-heal phase starting on A100 $MACHINE_ID — retrying D3b D3c C2 C3 + installing mamba-ssm for B3_mamba. Will pause when done." start

# ── Step 1: Retry D3b + D3c TopoLoss (run.py now has loss_fn.to(device) fix) ──
echo "[final] === Retrying D3b (TopoLoss λ=0.001) ==="
run_expt D3b || true

echo "[final] === Retrying D3c (TopoLoss λ=0.01) ==="
run_expt D3c || true

slack "D3b+D3c retried with TopoLoss device fix" ok

# ── Step 2: Retry C2 Phi-4-MM (backoff installed earlier) ────────────────────
echo "[final] === Retrying C2 Phi-4-MM ==="
run_expt C2 || true

# ── Step 3: Retry C3 Whisper-Qwen (same device fix in run.py may help) ───────
echo "[final] === Retrying C3 Whisper-Qwen ==="
run_expt C3 || true

# ── Step 3b: Install mamba-ssm and run B3_mamba ──────────────────────────────
# Root cause: CUDA 12.6 toolkit vs PyTorch+cu130.
# Fix: install CUDA 13.0 toolkit so nvcc matches torch's compiled CUDA version.
echo "[final] === Installing CUDA 13.0 toolkit for mamba-ssm ==="
slack "Installing CUDA 13.0 toolkit for B3_mamba..." ok

cuda130_ok=false

# Try apt-get first (fastest)
if apt-get install -y cuda-toolkit-13-0 2>/dev/null; then
    export CUDA_HOME=/usr/local/cuda-13.0
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    echo "[final] CUDA 13.0 installed via apt-get."
    cuda130_ok=true
else
    echo "[final] apt-get for cuda-toolkit-13-0 failed. Trying conda..."
    # Fallback: conda install matching cuda toolkit
    if conda install -y -c nvidia cuda-toolkit=13.0 2>/dev/null; then
        export CUDA_HOME=$(conda info --base)/pkgs/cuda-toolkit*/
        echo "[final] CUDA 13.0 installed via conda."
        cuda130_ok=true
    else
        echo "[final] conda also failed. Trying TORCH_CUDA_ARCH_LIST override..."
        # Last resort: tell the compiler to build for SM 8.0 (A100)
        # and skip the strict version check via env override
        export TORCH_CUDA_ARCH_LIST="8.0"
    fi
fi

# Install mamba-ssm regardless (will use whatever nvcc is available)
echo "[final] Installing mamba-ssm causal-conv1d..."
pip install mamba-ssm causal-conv1d --no-build-isolation -q 2>/tmp/mamba_build.log
mamba_rc=$?
if [ $mamba_rc -eq 0 ]; then
    echo "[final] mamba-ssm installed successfully."
    slack "mamba-ssm installed. Running B3_mamba now." ok

    echo "[final] === B3_mamba (true Mamba SSM encoder) ==="
    run_expt B3_mamba || true
else
    echo "[final] mamba-ssm build FAILED (exit $mamba_rc). Skipping B3_mamba."
    tail -20 /tmp/mamba_build.log >> /home/final.log || true
    slack "B3_mamba SKIPPED — mamba-ssm build failed even after CUDA 13.0 attempt. See /tmp/mamba_build.log." fail
fi

# ── Step 4: Full final leaderboard ───────────────────────────────────────────
echo "[final] === FINAL LEADERBOARD ==="
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs WHERE profile='toy' ORDER BY slope ASC, expt_id\").fetchall()
c.close()
seen = {}
for r in rows:
    if r[0] not in seen: seen[r[0]] = r

print('Expt               WER@10      slope   best_wer  status')
print('-'*62)
for k in sorted(seen.keys(), key=lambda x: (seen[x][2] is None, seen[x][2] or 99)):
    r = seen[k]
    w = f'{r[1]:.4f}' if r[1] is not None else 'N/A   '
    s = f'{r[2]:.5f}' if r[2] is not None else 'N/A      '
    b = 'inf' if r[3] == float('inf') else (f'{r[3]:.4f}' if r[3] else 'N/A')
    st = 'DONE' if r[1] is not None else 'FAIL'
    print(f'{r[0]:<19} {w:>8}  {s:>10}  {b:>8}  {st}')
" 2>/dev/null | tee -a /home/final.log || true

# ── Step 5: Slack the final summary ──────────────────────────────────────────
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')

def best(track_prefix, by='slope'):
    rows = c.execute(f\"SELECT expt_id, wer_at_ep10, slope FROM runs WHERE expt_id LIKE '{track_prefix}%' AND profile='toy' AND wer_at_ep10 IS NOT NULL ORDER BY slope ASC LIMIT 1\").fetchone()
    return rows

b_best = best('B')
d_best = best('D')
e_best = best('E')
c_best = best('C')
c.close()

lines = ['Sweep complete — BEST BUILDING BLOCKS:']
if b_best: lines.append(f'  Encoder:   {b_best[0]} slope={b_best[2]:.4f} WER@10={b_best[1]:.4f}')
if d_best: lines.append(f'  Loss:      {d_best[0]} slope={d_best[2]:.4f} WER@10={d_best[1]:.4f}')
if e_best: lines.append(f'  Projector: {e_best[0]} slope={e_best[2]:.4f} WER@10={e_best[1]:.4f}')
if c_best: lines.append(f'  Decoder:   {c_best[0]} slope={c_best[2]:.4f} WER@10={c_best[1]:.4f}')
else: lines.append('  Decoder:   C-track failed (C1/C3 device bugs, C2 retry done)')
print('\n'.join(lines))
" 2>/dev/null | while IFS= read -r line; do
    echo "[final] $line"
done

# Post the summary to Slack
summary=$(python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
b = c.execute(\"SELECT expt_id,slope FROM runs WHERE expt_id LIKE 'B%' AND wer_at_ep10 IS NOT NULL ORDER BY slope LIMIT 1\").fetchone()
d = c.execute(\"SELECT expt_id,slope FROM runs WHERE expt_id LIKE 'D%' AND wer_at_ep10 IS NOT NULL ORDER BY slope LIMIT 1\").fetchone()
e = c.execute(\"SELECT expt_id,slope FROM runs WHERE expt_id LIKE 'E%' AND wer_at_ep10 IS NOT NULL ORDER BY slope LIMIT 1\").fetchone()
c.close()
enc = f'{b[0]}(slope={b[1]:.4f})' if b else 'none'
loss = f'{d[0]}(slope={d[1]:.4f})' if d else 'none'
proj = f'{e[0]}(slope={e[1]:.4f})' if e else 'none'
print(f'SWEEP DONE. Best: encoder={enc} loss={loss} projector={proj}. B3_mamba attempted. Pausing instance now.')
" 2>/dev/null || echo "Sweep complete. Pausing instance.")
slack "$summary" ok

# ── Step 6: Guaranteed auto-pause ─────────────────────────────────────────────
# Retry up to 3 times with 10s gap
for attempt in 1 2 3; do
    echo "[final] Pause attempt $attempt/3..."
    pause_instance
    # Verify pause by checking instance state
    sleep 10
    state=$(curl -s -H "Authorization: Bearer $JL_TOKEN" \
        "$JL_API/users/fetch/$MACHINE_ID" \
        --max-time 15 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('instance',{}).get('status','?'))" 2>/dev/null || echo "?")
    echo "[final] Instance state after pause attempt $attempt: $state"
    if [ "$state" = "Pausing" ] || [ "$state" = "Paused" ]; then
        echo "[final] Pause confirmed: $state"
        break
    fi
    sleep 10
done

echo "[final] final.sh COMPLETE at $(date -u)"
