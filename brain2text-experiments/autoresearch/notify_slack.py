#!/usr/bin/env python3
"""
autoresearch/notify_slack.py — minimal Slack webhook notifier for sweep events.

Reads SLACK_WEBHOOK_URL from the environment. NEVER raises: a Slack or network
failure must never break a training sweep — it prints a warning and returns False.

Setup (once, on the A100 box):
    export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/XXX/YYY/ZZZ"

CLI:
    python autoresearch/notify_slack.py "B1 done — WER@10 0.41, slope +7% -> PROMOTE" --status ok
    python autoresearch/notify_slack.py "B2 OOM after 2 heal attempts — deferred"   --status fail

Import:
    from autoresearch.notify_slack import notify
    notify("sweep complete", status="ok")
"""
import argparse
import json
import os
import urllib.request

_EMOJI = {
    "ok":    ":white_check_mark:",
    "warn":  ":warning:",
    "fail":  ":x:",
    "info":  ":information_source:",
    "start": ":rocket:",
    "pause": ":zzz:",
    "heal":  ":wrench:",
}


def notify(message: str, status: str = "info") -> bool:
    """Post `message` to the Slack webhook. Returns True on HTTP 200, else False.
    Silent (returns False) if SLACK_WEBHOOK_URL is unset. Never raises."""
    url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not url:
        print(f"[slack] SLACK_WEBHOOK_URL not set — skipping: {message}")
        return False

    prefix = _EMOJI.get(status, "")
    payload = {"text": f"{prefix} {message}".strip()}
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = (resp.status == 200)
        if not ok:
            print(f"[slack] non-200 response: {resp.status}")
        return ok
    except Exception as exc:           # never let Slack break a sweep
        print(f"[slack] notify failed (ignored): {exc}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Post a message to Slack.")
    p.add_argument("message")
    p.add_argument("--status", default="info", choices=list(_EMOJI))
    args = p.parse_args()
    notify(args.message, args.status)


if __name__ == "__main__":
    main()
