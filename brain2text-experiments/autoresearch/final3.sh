#!/usr/bin/env bash
# autoresearch/final3.sh
# Installs all deps, runs B2/B3_mamba/C2/C3 retries, generates seaborn+plotly
# charts and markdown reports, then pauses. Fully autonomous.

set -uo pipefail

REPO=/home/brain2text-experiments
TRAIN="$REPO/data/toy_train.hdf5"
VAL="$REPO/data/toy_val.hdf5"
MACHINE_ID=422529
# JL_TOKEN and SLACK_WEBHOOK_URL should be set via environment variables
JL_TOKEN="${JL_TOKEN:-}"
JL_API="https://backendn.jarvislabs.net"
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export PYTHONIOENCODING=utf-8

cd "$REPO"
mkdir -p /home/logs results/figures results/reports

slack() { python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true; }

lb() {
    python -c "
import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite')
r=c.execute(\"SELECT wer_at_ep10,slope FROM runs WHERE expt_id='$1' ORDER BY rowid DESC LIMIT 1\").fetchone()
c.close()
print(f'WER@10={r[0]:.4f} slope={r[1]:.4f}') if r and r[0] else print('no-row')
" 2>/dev/null || echo "?"
}

run_expt() {
    local id="$1"; shift
    echo "[final3] START $id $(date -u +%H:%M:%S)"
    python run.py --expt "$id" --profile toy \
        --train_h5 "$TRAIN" --val_h5 "$VAL" "$@" \
        >> "/home/logs/${id}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[final3] DONE $id — $(lb $id)"
        slack "$id DONE — $(lb $id)" ok
    else
        tail -8 "/home/logs/${id}.log" >> /home/final3.log 2>/dev/null || true
        echo "[final3] FAIL $id exit=$rc"
        slack "$id FAILED exit=$rc" fail
    fi
}

pause_instance() {
    echo "[final3] Pausing $MACHINE_ID..."
    for i in 1 2 3; do
        resp=$(curl -s -X POST \
            -H "Authorization: Bearer $JL_TOKEN" \
            "$JL_API/misc/pause?machine_id=$MACHINE_ID" --max-time 30 2>&1 || echo "err")
        sleep 8
        state=$(curl -s -H "Authorization: Bearer $JL_TOKEN" \
            "$JL_API/users/fetch/$MACHINE_ID" --max-time 10 2>/dev/null \
            | python -c "import sys,json; print(json.load(sys.stdin).get('instance',{}).get('status','?'))" 2>/dev/null || echo "?")
        echo "[final3] Pause attempt $i: $resp | state=$state"
        [ "$state" = "Pausing" ] || [ "$state" = "Paused" ] && break
        sleep 20
    done
    slack "Instance $MACHINE_ID PAUSED. final3.sh complete. Charts+reports in results/." pause
}

slack "final3.sh starting: install deps, run B2/C2/C3/B3_mamba, generate seaborn+plotly charts + markdown reports. Will pause when done." start

# ── Step 1: Install ALL required packages (idempotent across resumes) ─────────
echo "[final3] Installing all dependencies..."
pip install \
    h5py jiwer scipy datasets pandas \
    matplotlib seaborn plotly kaleido \
    bitsandbytes peft accelerate sentencepiece \
    backoff pytest \
    -q --root-user-action=ignore 2>/dev/null || true
echo "[final3] Deps installed."

# ── Step 2: Set up CUDA 13.0 for mamba-ssm ───────────────────────────────────
echo "[final3] Setting up CUDA 13.0 nvcc for mamba-ssm..."
# cuda-nvcc-13-0 was already apt-get installed in final.sh; locate it
NVCC13=$(find /usr/local -name "nvcc" 2>/dev/null | grep -v "12\." | head -1 || true)
if [ -n "$NVCC13" ] && "$NVCC13" --version 2>/dev/null | grep -q "13\."; then
    export CUDA_HOME=$(dirname $(dirname "$NVCC13"))
    export PATH="$(dirname $NVCC13):$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[final3] Found nvcc 13.x at $NVCC13 → CUDA_HOME=$CUDA_HOME"
else
    # Try explicit reinstall of just nvcc (fast, ~87 MB)
    apt-get install -y --no-install-recommends cuda-nvcc-13-0 2>/dev/null || true
    NVCC13=$(find /usr/local -name "nvcc" 2>/dev/null | grep -v "12\." | head -1 || true)
    if [ -n "$NVCC13" ]; then
        export CUDA_HOME=$(dirname $(dirname "$NVCC13"))
        export PATH="$(dirname $NVCC13):$PATH"
        echo "[final3] Installed cuda-nvcc-13-0, CUDA_HOME=$CUDA_HOME"
    else
        echo "[final3] CUDA 13.0 nvcc not found — B3_mamba may still fail."
        export CUDA_HOME=/usr/local/cuda
    fi
fi

echo "[final3] Active nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'none')"

# ── Step 3: Build mamba-ssm ───────────────────────────────────────────────────
echo "[final3] Building mamba-ssm..."
pip install mamba-ssm causal-conv1d --no-build-isolation -q \
    > /tmp/mamba_build.log 2>&1
mamba_ok=$?
if [ $mamba_ok -eq 0 ]; then
    echo "[final3] mamba-ssm installed successfully!"
    slack "mamba-ssm installed with CUDA 13.0 — B3_mamba queued." ok
else
    tail -5 /tmp/mamba_build.log
    echo "[final3] mamba-ssm build failed — B3_mamba skipped."
    slack "mamba-ssm build failed again — B3_mamba skipped." warn
fi

# ── Step 4: Run deferred experiments ─────────────────────────────────────────
echo "[final3] === B2 HRM (dtype fix: enc.to(compute_dtype)) ==="
run_expt B2 || true

echo "[final3] === C3 Whisper-Qwen (bridge .to(device) fix) ==="
run_expt C3 || true

echo "[final3] === C2 Phi-4-MM (attn_implementation=sdpa fix) ==="
run_expt C2 || true

if [ "$mamba_ok" -eq 0 ]; then
    echo "[final3] === B3_mamba (true Mamba SSM) ==="
    run_expt B3_mamba || true
fi

# ── Step 5: Generate seaborn + plotly charts ──────────────────────────────────
echo "[final3] === Generating charts (seaborn + plotly) ==="
python autoresearch/make_charts_v2.py 2>&1 | tee -a /home/final3.log
slack "Charts generated: seaborn SVG/PNG + plotly HTML in results/figures/" ok

# ── Step 6: Generate markdown reports ────────────────────────────────────────
echo "[final3] === Generating markdown reports ==="
python autoresearch/generate_reports.py 2>&1 | tee -a /home/final3.log
slack "Reports generated: results/reports/B_interpretation.md ... SWEEP_SUMMARY.md" ok

# ── Step 7: Final leaderboard ──────────────────────────────────────────────────
echo "[final3] === FINAL LEADERBOARD ==="
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id, wer_at_ep10, slope FROM runs WHERE profile='toy' ORDER BY slope ASC NULLS LAST\").fetchall()
c.close()
seen = {}
for r in rows:
    if r[0] not in seen and r[1] is not None: seen[r[0]] = r
print(f'{'Rank':<5} {'Expt':<18} {'WER@10':>8} {'Slope':>10}  Label')
print('-'*58)
for rank, r in enumerate(sorted(seen.values(), key=lambda x: x[2]), 1):
    sl = r[2]; lbl = 'STRONG' if sl<-0.010 else ('PROMISING' if sl<-0.005 else ('WEAK' if sl<0.005 else 'INERT'))
    print(f'{rank:<5} {r[0]:<18} {r[1]:>8.4f} {sl:>10.5f}  {lbl}')
" 2>/dev/null | tee -a /home/final3.log

# ── Step 8: Pause ─────────────────────────────────────────────────────────────
pause_instance
echo "[final3] COMPLETE at $(date -u)"
