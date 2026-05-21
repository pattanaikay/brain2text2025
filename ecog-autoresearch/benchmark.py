from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from train import TrainConfig, train_model


RESULT_FIELDS = [
    "timestamp",
    "run_id",
    "model",
    "seed",
    "subject",
    "budget_minutes",
    "epochs",
    "steps",
    "metric",
    "mean_pearson",
    "rmse",
    "params",
    "peak_vram_mb",
    "train_seconds",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed ECoG autoresearch benchmark runner.")
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--data-path", default="data/ecog_fingerflex_subject1.npz")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--budget-minutes", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--results-path", default="results.tsv")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def ensure_results_header(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writeheader()


def append_result(path: Path, row: dict[str, object]) -> None:
    ensure_results_header(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{args.model}_s{args.seed}"
    run_dir = Path(args.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        model=args.model,
        data_path=args.data_path,
        run_dir=str(run_dir),
        seed=args.seed,
        subject=args.subject,
        budget_minutes=args.budget_minutes,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
    )

    status = "ok"
    try:
        result = train_model(config)
    except Exception as exc:
        status = "failed"
        result = {
            "model": args.model,
            "seed": args.seed,
            "subject": args.subject,
            "budget_minutes": args.budget_minutes,
            "epochs": 0,
            "steps": 0,
            "metric": "mean_pearson",
            "mean_pearson": float("nan"),
            "rmse": float("nan"),
            "params": 0,
            "peak_vram_mb": 0.0,
            "train_seconds": 0.0,
            "error": repr(exc),
        }

    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps({"run_id": run_id, "config": asdict(config), "status": status, **result}, indent=2),
        encoding="utf-8",
    )

    row = {
        "timestamp": timestamp,
        "run_id": run_id,
        "status": status,
        "notes": args.notes,
        **result,
    }
    append_result(Path(args.results_path), row)

    print(json.dumps({"run_id": run_id, "status": status, **result}, indent=2))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
