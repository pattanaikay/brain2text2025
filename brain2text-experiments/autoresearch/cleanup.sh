#!/usr/bin/env bash
set -uo pipefail
REPO=/home/brain2text-experiments
TRAIN="$REPO/data/toy_train.hdf5"
VAL="$REPO/data/toy_val.hdf5"
# SLACK_WEBHOOK_URL should be set via environment variable
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export PYTHONIOENCODING=utf-8
cd "$REPO"

slack() { python autoresearch/notify_slack.py "$1" --status "${2:-ok}" 2>/dev/null || true; }

# Wait for combo phase (run_remaining.sh / B1 runs) to finish
echo "[cleanup] Waiting for run_remaining.sh and combo runs to finish..."
while pgrep -f "run_remaining.sh" > /dev/null 2>&1 || pgrep -f "run.py.*--expt B1" > /dev/null 2>&1; do
    sleep 20
done
echo "[cleanup] Combo phase done at $(date -u)"

# Retry C2 Phi-4-MM (backoff now installed)
echo "[cleanup] Retrying C2 Phi-4-MM..."
python run.py --expt C2 --profile toy --train_h5 "$TRAIN" --val_h5 "$VAL" >> /home/logs/C2.log 2>&1
rc=$?
if [ $rc -eq 0 ]; then
    row=$(python -c "import sqlite3; c=sqlite3.connect('results/leaderboard.sqlite'); r=c.execute(\"SELECT wer_at_ep10,slope FROM runs WHERE expt_id='C2' ORDER BY rowid DESC LIMIT 1\").fetchone(); c.close(); print(f'WER@10={r[0]:.4f} slope={r[1]:.4f}') if r and r[0] else print('no-row')" 2>/dev/null || echo '?')
    echo "[cleanup] C2 DONE: $row"
    slack "C2 Phi-4-MM RETRY DONE — $row" ok
else
    echo "[cleanup] C2 FAILED again (exit $rc)"
    slack "C2 FAILED again exit=$rc — check /home/logs/C2.log" fail
fi

# Final full leaderboard
echo "[cleanup] === FINAL LEADERBOARD ==="
python -c "
import sqlite3
c = sqlite3.connect('results/leaderboard.sqlite')
rows = c.execute(\"SELECT expt_id, wer_at_ep10, slope, best_wer FROM runs WHERE profile='toy' ORDER BY expt_id, rowid DESC\").fetchall()
c.close()
seen = {}
for r in rows:
    if r[0] not in seen: seen[r[0]] = r
print('Expt               WER@10     slope   best_wer')
print('-'*52)
for k in sorted(seen.keys()):
    r = seen[k]
    w = str(round(r[1],4)) if r[1] is not None else 'N/A'
    s = str(round(r[2],5)) if r[2] is not None else 'N/A'
    b = str(round(r[3],4)) if r[3] is not None and r[3] != float('inf') else 'inf'
    print(f'{r[0]:<19} {w:>8}  {s:>9}  {b:>8}')
" 2>/dev/null || true

slack "Sweep complete. All tracks done. C2 retried. Check results/leaderboard.sqlite." ok
echo "[cleanup] ALL DONE at $(date -u)"
