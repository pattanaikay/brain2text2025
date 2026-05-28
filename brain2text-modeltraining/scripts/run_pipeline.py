"""
Brain2Text Pipeline Orchestrator
=================================
Runs SSL → CTC sequentially on the cluster with:
  - Real-time stdout monitoring and metric parsing
  - Notifications every N epochs via Discord, Slack, or Telegram
  - Health checks vs expected thresholds
  - OOM auto-repair: retries with halved batch_size (up to 3x)
  - Central auto-pause of the JarvisLabs instance when done

Notification setup (pick one):

  Discord  — any server → right-click channel → Edit Channel →
             Integrations → Webhooks → New Webhook → Copy URL
             Pass as: --webhook_url "https://discord.com/api/webhooks/..."

  Slack    — api.slack.com/apps → New App → Incoming Webhooks → Add to Workspace
             Pass as: --webhook_url "https://hooks.slack.com/services/..."

  Telegram — 1. Message @BotFather on Telegram → /newbot → copy the token
             2. Start a chat with your new bot (search its name → press Start)
             3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates → copy chat.id
             Pass as: --tg_token "123456:ABC..." --tg_chat_id "987654321"

Leave all notification args empty to print to stdout only.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Expected metric thresholds (from BIT paper + earlier analysis) ───────────
SSL_WARN_EPOCH       = 20      # check SSL health at this epoch
SSL_WARN_VAL_LOSS    = 0.40   # if val_loss > this at epoch 20, warn
CTC_WARN_EPOCH       = 50      # check CTC health at this epoch
CTC_WARN_PER         = 0.70   # if PER > this at epoch 50, warn
CTC_TARGET_PER       = 0.35   # "ready for E2E" threshold
E2E_TARGET_WER       = 0.10   # paper target

OOM_MAX_RETRIES      = 3
OOM_MIN_BATCH        = 8


# ── Notification helpers ──────────────────────────────────────────────────────

def ts():
    return datetime.now().strftime("%H:%M:%S")


def notify(body: str, title: str = "", webhook_url: str = "",
           tg_token: str = "", tg_chat_id: str = ""):
    """
    Send a notification via Discord, Slack, or Telegram.
    Auto-detects service from the webhook URL format.
    Always prints to stdout as well.
    """
    line = f"[{ts()}] {title}: {body}" if title else f"[{ts()}] {body}"
    print(line, flush=True)

    # ── Telegram ──────────────────────────────────────────────────────────────
    if tg_token and tg_chat_id:
        try:
            msg = f"*{title}*\n{body}" if title else body
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            requests.post(url, json={
                "chat_id": tg_chat_id,
                "text": msg,
                "parse_mode": "Markdown"
            }, timeout=10).raise_for_status()
        except Exception as e:
            print(f"[TELEGRAM ERROR] {e}", flush=True)
        return  # don't double-send if webhook also provided

    # ── Discord or Slack via webhook ──────────────────────────────────────────
    if not webhook_url:
        return
    try:
        msg = f"**{title}**\n{body}" if title else body
        if "hooks.slack.com" in webhook_url:
            # Slack uses "text" key; strip markdown bold for cleaner rendering
            payload = {"text": f"*{title}*\n{body}" if title else body}
        else:
            # Discord uses "content" key
            payload = {"content": msg}
        requests.post(webhook_url, json=payload, timeout=10).raise_for_status()
    except Exception as e:
        print(f"[WEBHOOK ERROR] {e}", flush=True)


def make_notifier(webhook_url: str, tg_token: str, tg_chat_id: str):
    """Returns a bound notify() callable so callers don't repeat args."""
    def _notify(body: str, title: str = ""):
        notify(body, title=title, webhook_url=webhook_url,
               tg_token=tg_token, tg_chat_id=tg_chat_id)
    return _notify


def pause_instance(instance_id: str, notifier):
    """Pause the JarvisLabs instance using the jl CLI (authenticated, reliable)."""
    if not instance_id:
        notifier("No instance_id provided — skipping auto-pause.", "⚠️ Auto-Pause Skipped")
        return
    notifier(f"Pausing instance {instance_id} via jl CLI...", "💤 Auto-Pause")
    try:
        result = subprocess.run(
            ["jl", "pause", str(instance_id), "--yes", "--json"],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            notifier(f"Instance {instance_id} paused successfully. Compute billing stopped.", "✅ Instance Paused")
        else:
            err = result.stderr.strip() or result.stdout.strip()
            notifier(f"jl pause exited {result.returncode}: {err}", "⚠️ Pause Warning")
    except FileNotFoundError:
        notifier("jl CLI not found — cannot auto-pause. Please pause manually.", "⚠️ Auto-Pause Failed")
    except Exception as e:
        notifier(f"Unexpected error during pause: {e}", "⚠️ Pause Error")


# ── Core: run one training stage ─────────────────────────────────────────────

def run_stage(cmd: list, stage: str, notifier, notify_interval: int = 10, sub_env: dict = None):
    """
    Stream stdout from the subprocess line-by-line.
    Parses SSL val_loss / CTC PER / E2E WER from log lines.
    Returns (success: bool, last_metric: dict).

    OOM recovery: if 'CUDA out of memory' appears in output, halves the
    --batch_size argument and retries from the last checkpoint (up to
    OOM_MAX_RETRIES times).
    """

    def find_batch_idx(c):
        for i, tok in enumerate(c):
            if tok == "--batch_size":
                return i + 1
        return None

    oom_retries = 0
    last_metric: dict = {}

    while True:
        notifier(f"`{' '.join(cmd)}`",
                 f"🚀 {stage} — starting (OOM attempt {oom_retries + 1}/{OOM_MAX_RETRIES + 1})")

        oom_flag = False
        last_notify_epoch = -notify_interval

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=sub_env,  # None = inherit parent env unchanged
        )

        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            print(line, flush=True)

            # ── OOM detection ─────────────────────────────────────────────
            if "out of memory" in line.lower():
                oom_flag = True
                continue

            # ── Fatal errors that cannot be auto-repaired ─────────────────
            if "All trials were filtered out" in line:
                notifier("CTC filter removed ALL trials. patch_size is too large for your "
                         "phoneme sequences. Reduce --patch_size and redeploy.",
                         f"💀 {stage} — Fatal")
                proc.terminate()
                return False, last_metric

            if "SMOKE TEST FAILED" in line:
                notifier(f"E2E smoke test failed before training started.\n{line}",
                         f"💀 {stage} — Smoke Test Failed")
                proc.terminate()
                return False, last_metric

            if "TRAINING FATAL ERROR" in line:
                notifier(line, f"💀 {stage} — Fatal Error")
                proc.terminate()
                return False, last_metric

            # ── Metric parsing ────────────────────────────────────────────
            epoch_m = re.search(r'[Ee]poch[^\d]*(\d+)', line)
            current_epoch = int(epoch_m.group(1)) if epoch_m else None

            # SSL: "Epoch N val_loss: X.XXXX"
            ssl_m = re.search(r'val_loss[:\s]+([\d.naninf]+)', line)
            if ssl_m and current_epoch:
                try:
                    val_loss = float(ssl_m.group(1))
                except ValueError:
                    val_loss = float("nan")
                last_metric = {"epoch": current_epoch, "val_loss": val_loss}

                if val_loss != val_loss:  # NaN
                    notifier(f"NaN val_loss at epoch {current_epoch}. Training diverged.",
                             f"💀 {stage} — Diverged")
                    proc.terminate()
                    return False, last_metric

                if current_epoch == SSL_WARN_EPOCH and val_loss > SSL_WARN_VAL_LOSS:
                    notifier(f"val_loss={val_loss:.4f} at epoch {current_epoch} "
                             f"(expected < {SSL_WARN_VAL_LOSS}). "
                             "Check session_stats.json loaded correctly.",
                             f"⚠️ {stage} — Health Warning")

            # CTC: "Validation Epoch N: PER=X.XXXX"
            ctc_m = re.search(r'PER[=:\s]+([\d.naninf]+)', line)
            if ctc_m and current_epoch:
                try:
                    per = float(ctc_m.group(1))
                except ValueError:
                    per = float("nan")
                last_metric = {"epoch": current_epoch, "per": per}

                if current_epoch == CTC_WARN_EPOCH and per > CTC_WARN_PER:
                    notifier(f"PER={per:.4f} at epoch {current_epoch} "
                             f"(expected < {CTC_WARN_PER}). "
                             "SSL encoder may not have loaded — check --ssl_checkpoint path.",
                             f"⚠️ {stage} — Health Warning")

                if per < CTC_TARGET_PER:
                    notifier(f"PER={per:.4f} < target {CTC_TARGET_PER} at epoch {current_epoch}.",
                             f"🎯 {stage} — CTC Target Hit")

            # E2E: "Validation Epoch N: WER=X.XXXX"
            e2e_m = re.search(r'WER[=:\s]+([\d.naninf]+)', line)
            if e2e_m and current_epoch:
                try:
                    wer = float(e2e_m.group(1))
                except ValueError:
                    wer = float("nan")
                last_metric = {"epoch": current_epoch, "wer": wer}

                if wer < E2E_TARGET_WER:
                    notifier(f"WER={wer:.4f} < target {E2E_TARGET_WER} at epoch {current_epoch}!",
                             f"🏆 {stage} — TARGET ACHIEVED")

            # ── Periodic notification ─────────────────────────────────────
            if current_epoch is not None and (current_epoch - last_notify_epoch) >= notify_interval:
                if "val_loss" in last_metric:
                    summary = f"Epoch {current_epoch} | val_loss={last_metric['val_loss']:.4f}"
                elif "per" in last_metric:
                    summary = f"Epoch {current_epoch} | PER={last_metric['per']:.4f}"
                elif "wer" in last_metric:
                    summary = f"Epoch {current_epoch} | WER={last_metric['wer']:.4f}"
                else:
                    summary = f"Epoch {current_epoch} | training..."
                notifier(summary, f"📊 {stage} Update")
                last_notify_epoch = current_epoch

        proc.wait()

        # ── OOM recovery ──────────────────────────────────────────────────
        if oom_flag:
            if oom_retries >= OOM_MAX_RETRIES:
                notifier(f"OOM after {oom_retries} retries. Minimum batch_size={OOM_MIN_BATCH} reached.",
                         f"💀 {stage} — OOM Unrecoverable")
                return False, last_metric

            bidx = find_batch_idx(cmd)
            if bidx is None:
                notifier("OOM detected but --batch_size not in command.", f"💀 {stage} — OOM")
                return False, last_metric

            current_bs = int(cmd[bidx])
            new_bs = max(OOM_MIN_BATCH, current_bs // 2)
            if new_bs == current_bs:
                notifier(f"OOM at minimum batch_size={current_bs}. Cannot reduce further.",
                         f"💀 {stage} — OOM")
                return False, last_metric

            cmd[bidx] = str(new_bs)
            oom_retries += 1
            notifier(f"OOM! Retrying with batch_size={new_bs} "
                     f"(attempt {oom_retries}/{OOM_MAX_RETRIES}). "
                     "Checkpoint will resume from last saved epoch.",
                     f"⚠️ {stage} — OOM → Retry")
            time.sleep(5)
            continue

        # ── Clean exit ────────────────────────────────────────────────────
        if proc.returncode == 0:
            notifier(f"Final metric: {json.dumps(last_metric)}", f"✅ {stage} — Complete")
            return True, last_metric
        else:
            notifier(f"Process exited with code {proc.returncode}. Check logs.",
                     f"❌ {stage} — Non-zero Exit")
            return False, last_metric


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    notifier = make_notifier(args.webhook_url, args.tg_token, args.tg_chat_id)

    notifier(f"Data: {args.data_dir}\n"
             f"Output: {args.output_dir}\n"
             f"Stages: SSL ({args.ssl_epochs} epochs) → CTC ({args.ctc_epochs} epochs)\n"
             f"Notify every: {args.notify_interval} epochs",
             "🧠 Brain2Text Pipeline Starting")

    python_bin   = args.python_bin
    scripts_dir  = args.scripts_dir
    extra_pypath = args.extra_pythonpath

    # Build subprocess env — prepend extra PYTHONPATH entries
    sub_env = os.environ.copy()
    if extra_pypath:
        sub_env["PYTHONPATH"] = extra_pypath + ":" + sub_env.get("PYTHONPATH", "")

    # ── Stage 1: SSL ──────────────────────────────────────────────────────────
    ssl_out = f"{args.output_dir}/ssl"
    ssl_cmd = [
        python_bin, f"{scripts_dir}/train_ssl.py",
        "--train_h5",      args.data_dir,
        "--val_h5",        args.data_dir,
        "--output_dir",    ssl_out,
        "--session_stats", args.session_stats,
        "--epochs",        str(args.ssl_epochs),
        "--batch_size",    str(args.ssl_batch_size),
        "--lr",            "1e-4",
        "--patch_size",    str(args.patch_size),
        "--num_workers",   str(args.num_workers),
        "--no_autopause",
    ]

    ssl_ok, ssl_metric = run_stage(ssl_cmd, "SSL Pretraining", notifier, args.notify_interval,
                                   sub_env=sub_env)

    if not ssl_ok:
        notifier("SSL failed. Aborting pipeline.", "❌ Pipeline Aborted")
        pause_instance(args.instance_id, notifier)
        sys.exit(1)

    ssl_ckpt = f"{ssl_out}/best_encoder_ssl.pth"
    if not os.path.exists(ssl_ckpt):
        notifier(f"SSL reported success but {ssl_ckpt} not found. Cannot start CTC.",
                 "❌ Pipeline Aborted — Missing SSL Checkpoint")
        pause_instance(args.instance_id, notifier)
        sys.exit(1)

    # ── Stage 2: CTC ──────────────────────────────────────────────────────────
    ctc_out = f"{args.output_dir}/ctc"
    ctc_cmd = [
        python_bin, f"{scripts_dir}/train_ctc.py",
        "--train_h5",       args.data_dir,
        "--val_h5",         args.data_dir,
        "--output_dir",     ctc_out,
        "--ssl_checkpoint", ssl_ckpt,
        "--session_stats",  args.session_stats,
        "--epochs",         str(args.ctc_epochs),
        "--batch_size",     str(args.ctc_batch_size),
        "--lr",             "1e-4",
        "--patch_size",     str(args.patch_size),
        "--num_workers",    str(args.num_workers),
        "--no_autopause",
    ]

    ctc_ok, ctc_metric = run_stage(ctc_cmd, "CTC Fine-tuning", notifier, args.notify_interval,
                                   sub_env=sub_env)

    if not ctc_ok:
        notifier("CTC failed. Pausing instance.", "❌ Pipeline Aborted")
        pause_instance(args.instance_id, notifier)
        sys.exit(1)

    # ── Final summary ─────────────────────────────────────────────────────────
    ssl_loss_str = f"{ssl_metric.get('val_loss', 'N/A'):.4f}" if isinstance(ssl_metric.get('val_loss'), float) else "N/A"
    ctc_per  = ctc_metric.get("per")
    per_str  = f"{ctc_per:.4f}" if isinstance(ctc_per, float) else "N/A"
    e2e_ready = isinstance(ctc_per, float) and ctc_per < CTC_TARGET_PER
    e2e_line  = (f"✅ CTC PER={per_str} < {CTC_TARGET_PER} — ready for E2E."
                 if e2e_ready
                 else f"⚠️ CTC PER={per_str} >= {CTC_TARGET_PER} — review before E2E.")

    notifier(f"SSL final val_loss: {ssl_loss_str}\n"
             f"CTC final PER:      {per_str}\n"
             f"{e2e_line}\n\n"
             f"E2E command:\n"
             f"python scripts/train_e2e.py"
             f" --pretrained_encoder {ctc_out}/best_model_per.pth"
             f" --train_h5 {args.data_dir} --val_h5 {args.data_dir}"
             f" --output_dir {args.output_dir}/e2e"
             f" --session_stats {args.session_stats}"
             f" --epochs 80 --batch_size 4 --accumulation_steps 4"
             f" --patch_size 4 --val_interval 5",
             "🎉 Pipeline Complete")

    pause_instance(args.instance_id, notifier)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain2Text SSL→CTC pipeline orchestrator")
    parser.add_argument("--data_dir",        required=True,  help="Root dir with session subdirs containing data_train/val.hdf5")
    parser.add_argument("--output_dir",      default="/home/outputs")
    parser.add_argument("--session_stats",   required=True,  help="Path to session_stats.json")
    parser.add_argument("--instance_id",       default="410271", help="JarvisLabs instance ID for auto-pause")
    # Cluster paths
    parser.add_argument("--python_bin",        default=sys.executable, help="Python interpreter to use")
    parser.add_argument("--scripts_dir",       default=str(Path(__file__).parent), help="Directory containing train_ssl/ctc/e2e.py")
    parser.add_argument("--extra_pythonpath",  default="", help="Colon-separated paths to prepend to PYTHONPATH in subprocesses")
    parser.add_argument("--patch_size",        type=int, default=4, help="Patch size — must match across SSL/CTC/E2E")
    # Notification (pick one)
    parser.add_argument("--webhook_url",     default="", help="Discord webhook (discord.com/api/webhooks/...) or Slack webhook (hooks.slack.com/...)")
    parser.add_argument("--tg_token",        default="", help="Telegram bot token from @BotFather")
    parser.add_argument("--tg_chat_id",      default="", help="Telegram chat ID (get from api.telegram.org/bot<TOKEN>/getUpdates)")
    # Training
    parser.add_argument("--ssl_epochs",      type=int, default=50)
    parser.add_argument("--ssl_batch_size",  type=int, default=32)
    parser.add_argument("--ctc_epochs",      type=int, default=200)
    parser.add_argument("--ctc_batch_size",  type=int, default=64)
    parser.add_argument("--num_workers",     type=int, default=8)
    parser.add_argument("--notify_interval", type=int, default=10, help="Notify every N epochs")
    args = parser.parse_args()
    main(args)
