"""
run.py
------
Single entrypoint for all brain2text-experiments.

Usage:
    # Toy run (local, ~20 min):
    python run.py --expt B1 --profile toy \\
        --train_h5 data/toy_train.hdf5 --val_h5 data/val.hdf5

    # Full cloud run (A100, auto-pauses after):
    python run.py --expt B1 --profile full \\
        --train_h5 data/ --val_h5 data/

    # Analysis experiment (no training):
    python run.py --expt A1 --val_h5 data/val.hdf5

Responsibilities:
  1. Load registry.yaml + spec YAML
  2. Apply profile overrides (toy/full)
  3. Assert toy PASSED before allowing full run
  4. Write run.lock (code+spec hash)
  5. Build Stack from spec
  6. Wire CTC head and TopoLoss if needed
  7. Run smoke_assert at step 50
  8. Train with composed loss
  9. Log WER@10, slope, best WER to leaderboard.sqlite
 10. Auto-pause JarvisLabs instance if profile=full
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:
    import subprocess, sys as _sys
    subprocess.check_call([_sys.executable, "-m", "pip", "install", "pyyaml", "-q"])
    import yaml

# ── Path setup ─────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from stack import Stack
from compose import compose_from_spec
from tools.smoke_assert import SmokeAssert, SmokeAssertionError
from tools.tokenizer_wer import TokenizerWER
from docks.multiarch_dock import (
    Preprocessed_BCI_Dataset, bci_collate_fn,
    calculate_wer, calculate_cer, setup_logging,
)
from results.leaderboard import record_run, toy_passed_recently


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _spec_hash(spec: dict) -> str:
    canonical = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _write_run_lock(out_dir: str, spec: dict, profile: str, expt_id: str):
    import subprocess
    lock = {
        "expt_id":    expt_id,
        "profile":    profile,
        "spec_hash":  _spec_hash(spec),
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python":     sys.version,
    }
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_HERE), stderr=subprocess.DEVNULL,
        ).decode().strip()
        lock["git_sha"] = git_sha
    except Exception:
        lock["git_sha"] = "unknown"
    with open(os.path.join(out_dir, "run.lock"), "w") as f:
        json.dump(lock, f, indent=2)
    return lock


def _find_h5(root: str, pattern: str) -> list[str]:
    if os.path.isfile(root):
        return [root]
    return sorted(glob.glob(os.path.join(root, f"**/{pattern}"), recursive=True))


def _validate(model_stack, val_loader, device, compute_dtype, composed_loss,
               logger=None, max_batches=None) -> tuple[float, float, float]:
    """Run validation; returns (wer, cer, avg_loss)."""
    if not val_loader:
        return 1.0, 1.0, 0.0

    predictions, targets = [], []
    total_loss = 0
    enc = model_stack.encoder
    proj = model_stack.projector
    dec = model_stack.decoder

    enc.eval(); proj.eval(); dec.eval()
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
            if max_batches and bi >= max_batches:
                break
            neural = batch["neural"].to(device)
            lengths = batch["neural_lengths"].to(device)
            sid    = batch["session_id"]

            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                tokens = enc(neural, session_id=sid, neural_lengths=lengths)
                proj_out = proj(tokens)
                outputs = dec(proj_out, labels=batch["text"],
                               neural_lengths=lengths)
                total, breakdown = composed_loss(batch, model_stack, outputs)

            total_loss += total.item()
            try:
                preds = dec.generate(proj_out, neural_lengths=lengths)
            except Exception:
                preds = [""] * len(batch["text"])
            predictions.extend(preds)
            targets.extend(batch["text"])

    wer = calculate_wer(predictions, targets)
    cer = calculate_cer(predictions, targets)
    return wer, cer, total_loss / max(len(val_loader), 1)


# ── Main ────────────────────────────────────────────────────────────────────

def run(args):
    # 1. Load registry + spec
    registry = _load_yaml(_HERE / "registry.yaml")
    expt_map  = registry["experiments"]
    if args.expt not in expt_map:
        raise ValueError(
            f"Unknown experiment '{args.expt}'. "
            f"Valid IDs: {sorted(expt_map.keys())}"
        )
    expt_meta = expt_map[args.expt]
    spec_path = _HERE / expt_meta["spec_ref"]
    spec      = _load_yaml(spec_path)

    # 2. Apply profile overrides
    profile_path = _HERE / "profiles" / f"{args.profile}.yaml"
    profile      = _load_yaml(profile_path)
    spec         = _deep_merge(spec, {k: v for k, v in profile.items()
                                       if k not in ("profile", "smoke_assert",
                                                     "ranking", "auto_pause")})
    smoke_cfg    = profile.get("smoke_assert", {})
    ranking_cfg  = profile.get("ranking", {})
    auto_pause   = profile.get("auto_pause", False)
    instance_id  = profile.get("instance_id", "413754")

    # 3. Full-profile gate: require toy PASSED
    if args.profile == "full":
        if not toy_passed_recently(args.expt, days=7):
            raise RuntimeError(
                f"Experiment '{args.expt}' has no toy run PASSED in the last 7 days.\n"
                "Run with --profile toy first to validate the hypothesis locally."
            )

    # 4. Non-training experiments (A1, A2, A3)
    if not expt_meta.get("train_required", True):
        _run_analysis(args.expt, spec, args)
        return

    # 5. Output dir + run.lock
    run_dir = _HERE / "results" / "runs" / f"{args.expt}_{args.profile}_{_spec_hash(spec)}"
    os.makedirs(run_dir, exist_ok=True)
    logger = setup_logging(str(run_dir), log_name="run")
    lock   = _write_run_lock(str(run_dir), spec, args.profile, args.expt)
    logger.info(f"Run lock: {json.dumps(lock)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 6. Data
    train_h5s = _find_h5(args.train_h5, "data_train.hdf5")
    val_h5s   = _find_h5(args.val_h5,   "data_val.hdf5")
    if not train_h5s:
        raise FileNotFoundError(f"No training HDF5 found in {args.train_h5}")

    import h5py
    session_ids = set()
    for p in train_h5s:
        with h5py.File(p, "r") as f:
            for k in list(f.keys())[:1]:
                session_ids.add(str(f[k].attrs.get("session", "default")))
    spec.setdefault("encoder", {})["session_ids"] = list(session_ids)

    patch_size = spec.get("encoder", {}).get("patch_size", 4)
    train_ds   = Preprocessed_BCI_Dataset(train_h5s, patch_size=patch_size, augment=True)
    val_ds     = Preprocessed_BCI_Dataset(val_h5s,   patch_size=patch_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=spec.get("batch_size", 4),
                               shuffle=True, collate_fn=bci_collate_fn,
                               num_workers=spec.get("num_workers", 2), pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=spec.get("batch_size", 4),
                               shuffle=False, collate_fn=bci_collate_fn,
                               num_workers=spec.get("num_workers", 2), pin_memory=True)

    # 7. Build Stack
    logger.info(f"Building Stack for {args.expt}...")
    stack = Stack.from_spec(spec)
    logger.info(str(stack))

    enc  = stack.encoder.to(device)
    proj = stack.projector.to(device)
    dec  = stack.decoder  # decoder manages its own device_map

    # 8. Wire CTC head + TopoLoss if in loss spec
    composed_loss = compose_from_spec(spec.get("loss", [{"variant": "ce"}]))
    for loss_fn in composed_loss._fns:
        if hasattr(loss_fn, "attach"):   # TopoLossStage
            loss_fn.attach(enc)
        if hasattr(loss_fn, "ctc_head"):  # CTCAnnealLoss
            loss_fn.ctc_head = loss_fn.ctc_head.to(device)

    # 9. Optimizer
    trainable = (
        list(enc.parameters()) +
        list(proj.parameters()) +
        [p for p in dec.parameters() if p.requires_grad]
    )
    for fn in composed_loss._fns:
        if hasattr(fn, "parameters"):
            trainable.extend(fn.parameters())
    optimizer = AdamW(trainable, lr=spec.get("lr", 5e-5),
                      weight_decay=spec.get("weight_decay", 1e-5))
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # 10. Checkpoint resume
    ckpt_path = run_dir / "checkpoint_latest.pth"
    start_epoch = 1
    best_wer    = float("inf")
    history     = {"train_loss": [], "val_wer": [], "val_cer": [], "val_loss": []}

    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        # Smart load: skip shape mismatches
        for mod, key in [(enc, "encoder"), (proj, "projector")]:
            sd = {k[len(key)+1:]: v for k, v in ckpt.get("model_state_dict", {}).items()
                  if k.startswith(key+".")}
            mod.load_state_dict(sd, strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_wer    = ckpt.get("best_wer", float("inf"))
        history     = ckpt.get("history", history)
        logger.info(f"Resumed from epoch {start_epoch - 1}")

    # 11. Training loop
    epochs        = spec.get("epochs", 20)
    val_interval  = spec.get("val_interval", 2)
    max_batches   = spec.get("max_batches_per_epoch")
    accum_steps   = spec.get("accumulation_steps", 4)
    val_max_batch = spec.get("val_max_batches")
    patience      = spec.get("patience", 50)
    pat_counter   = 0
    ranking_wer_at10 = None

    smoke = SmokeAssert(smoke_cfg, check_at_step=50)

    try:
        for epoch in range(start_epoch, epochs + 1):
            enc.train(); proj.train()

            # Update annealed loss weights
            for fn in composed_loss._fns:
                if hasattr(fn, "set_epoch"):
                    fn.set_epoch(epoch)

            total_loss = 0.0
            optimizer.zero_grad()
            pbar = tqdm(train_loader, desc=f"Ep {epoch}/{epochs}")

            for step, batch in enumerate(pbar):
                if max_batches and step >= max_batches:
                    break

                neural  = batch["neural"].to(device)
                lengths = batch["neural_lengths"].to(device)
                sid     = batch["session_id"]

                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    tokens   = enc(neural, session_id=sid, neural_lengths=lengths)
                    proj_out = proj(tokens)
                    outputs  = dec(proj_out, labels=batch["text"],
                                    neural_lengths=lengths)
                    # Expose tensors for compose
                    outputs["neural_tokens"]   = tokens
                    outputs["projected_embeds"] = proj_out
                    outputs["batch"]           = batch

                    total, breakdown = composed_loss(batch, stack, outputs)
                    total = total / accum_steps

                total.backward()
                if (step + 1) % accum_steps == 0:
                    nn.utils.clip_grad_norm_(trainable, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += total.item() * accum_steps
                pbar.set_postfix({k: f"{v.item():.3f}" for k, v in breakdown.items()})

                # Smoke-assert check at step 50
                smoke.record(
                    step,
                    n_tokens = sum(len(t) for t in batch["text"]),
                    ce_loss  = breakdown.get("loss_ce", torch.tensor(0.0)).item(),
                    ctc_loss = breakdown.get("loss_ctc", torch.tensor(0.0)).item(),
                )

            avg_loss = total_loss / max(step + 1, 1)
            history["train_loss"].append(avg_loss)
            logger.info(f"Epoch {epoch} loss={avg_loss:.4f}")

            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "encoder_state_dict":   enc.state_dict(),
                "projector_state_dict": proj.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_wer": best_wer,
                "history":  history,
            }, str(ckpt_path))

            # Validation
            if epoch % val_interval == 0:
                wer, cer, vloss = _validate(
                    stack, val_loader, device, compute_dtype, composed_loss,
                    logger=logger, max_batches=val_max_batch,
                )
                logger.info(f"Val ep{epoch}: WER={wer:.4f} CER={cer:.4f} loss={vloss:.4f}")
                history["val_wer"].append(wer)
                history["val_cer"].append(cer)
                history["val_loss"].append(vloss)
                scheduler.step(wer)

                # Ranking metric: WER at epoch 10
                ep10 = ranking_cfg.get("wer_at_epoch", 10)
                if epoch == ep10:
                    ranking_wer_at10 = wer
                    logger.info(f"[Ranking] WER@{ep10} = {wer:.4f}")

                if wer < best_wer:
                    best_wer = wer
                    pat_counter = 0
                    torch.save(enc.state_dict(),  run_dir / "best_encoder.pth")
                    torch.save(proj.state_dict(), run_dir / "best_projector.pth")
                    logger.info(f"New best WER={wer:.4f}")
                else:
                    pat_counter += 1
                    if pat_counter >= patience:
                        logger.info("Early stopping.")
                        break

            with open(run_dir / "history.json", "w") as f:
                json.dump(history, f)

    except SmokeAssertionError as e:
        logger.error(f"SMOKE ASSERT FAILED:\n{e}")
        with open(run_dir / "smoke_fail.json", "w") as f:
            json.dump({"error": str(e)}, f)
        return

    except Exception as e:
        logger.error(f"TRAINING ERROR: {e}\n{traceback.format_exc()}")

    finally:
        # Compute slope for ranking
        wers = history["val_wer"]
        slope = None
        if len(wers) >= 2:
            slope = (wers[0] - wers[-1]) / max(len(wers) - 1, 1)

        # Log to leaderboard
        record_run(
            expt_id      = args.expt,
            profile      = args.profile,
            spec_hash    = _spec_hash(spec),
            best_wer     = best_wer,
            wer_at_ep10  = ranking_wer_at10,
            slope        = slope,
            run_dir      = str(run_dir),
        )

        # JarvisLabs auto-pause
        if auto_pause and args.profile == "full":
            try:
                import requests
                requests.get(f"https://jarvislabs.ai/pause/{instance_id}", timeout=5)
                logger.info(f"Auto-paused JarvisLabs instance {instance_id}")
            except Exception as pe:
                logger.warning(f"Auto-pause failed: {pe}")


def _run_analysis(expt_id: str, spec: dict, args):
    """Dispatch to the appropriate analysis tool."""
    dispatch = {
        "A1": "tools.cka_analysis",
        "A2": "tools.perplexity_test",
        "A3": "tools.phoneme_probe",
    }
    if expt_id not in dispatch:
        raise NotImplementedError(f"Analysis dispatch for {expt_id} not implemented")
    import importlib
    mod = importlib.import_module(dispatch[expt_id])
    # Each analysis tool has its own argparse; call main() with sys.argv untouched
    mod.main()


def main():
    parser = argparse.ArgumentParser(
        description="Brain2Text Experiment Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--expt",    required=True,  help="Experiment ID from registry.yaml (e.g. B1, D2a)")
    parser.add_argument("--profile", default="toy",  choices=["toy", "full"], help="Hardware profile")
    parser.add_argument("--train_h5", default=None,  help="Path to training HDF5 file or directory")
    parser.add_argument("--val_h5",   default=None,  help="Path to validation HDF5 file or directory")
    parser.add_argument("--override", nargs="*",     help="Key=value spec overrides, e.g. encoder.patch_size=5")
    args = parser.parse_args()

    # Apply CLI overrides to spec (handled inside run() via spec YAML)
    run(args)


if __name__ == "__main__":
    main()
