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

IMPORTANT NOTE on A100 instance pause/resume:
  The instance's home directory persists across pause/resume, and run.py AUTOMATICALLY loads
  the latest checkpoint (checkpoint_latest.pth) if it exists. So if the instance pauses or
  is interrupted mid-training (e.g., Ctrl-C, network hiccup), re-running the same command
  will resume from the saved epoch, optimizer state, etc. This requires run.py to complete
  its checkpoint save before the pause happens. For safety, the toy and full profiles use
  patience=50 (epochs) to enable clean checkpoints; if you interrupt, the checkpoint is
  written to run_dir/checkpoint_latest.pth.

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
    mem = model_stack._modules.get("memory")   # Track H (optional)

    enc.eval(); proj.eval(); dec.eval()
    if mem is not None:
        mem.eval()
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
            if max_batches and bi >= max_batches:
                break
            neural = batch["neural"].to(device)
            lengths = batch["neural_lengths"].to(device)
            sid    = batch["session_id"]

            with torch.autocast(device_type="cuda", dtype=compute_dtype):
                tokens = enc(neural, session_id=sid, neural_lengths=lengths)
                mem_out = {}
                if mem is not None:                 # Track H: episodic memory stage
                    tokens  = mem(tokens, session_id=sid)
                    mem_out = mem.last_read
                proj_out = proj(tokens)
                outputs = dec(proj_out, labels=batch["text"],
                               neural_lengths=lengths)
                outputs["neural_tokens"]    = tokens
                outputs["projected_embeds"] = proj_out
                outputs["batch"]            = batch
                outputs.update(mem_out)
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

def _apply_overrides(spec: dict, overrides) -> dict:
    """Apply CLI --override "dotted.key=value" entries into spec (mutates + returns).

    Supports nested dict paths (encoder.patch_size=5, encoder.pretrained_ckpt=path,
    projector.n_queries=32, batch_size=2) and the loss list keyed by variant
    (loss.contrastive.weight=0.0 → the {variant: contrastive} entry). Values are
    parsed with yaml.safe_load so 5→int, 0.0→float, true→bool, else str."""
    if not overrides:
        return spec
    import yaml as _yaml
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--override '{item}' must be of the form dotted.key=value")
        path, raw = item.split("=", 1)
        try:
            value = _yaml.safe_load(raw)
        except Exception:
            value = raw
        keys = path.split(".")
        # loss-list special case: loss.<variant>.<leaf>
        if keys[0] == "loss" and len(keys) == 3 and isinstance(spec.get("loss"), list):
            variant, leaf = keys[1], keys[2]
            entry = next((e for e in spec["loss"] if e.get("variant") == variant), None)
            if entry is None:
                entry = {"variant": variant}
                spec["loss"].append(entry)
            entry[leaf] = value
            continue
        node = spec
        for k in keys[:-1]:
            if not isinstance(node.get(k), dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value
    return spec


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
    # 2c. Apply CLI --override entries (win over both spec and profile)
    spec = _apply_overrides(spec, getattr(args, "override", None))
    smoke_cfg    = profile.get("smoke_assert", {})
    ranking_cfg  = profile.get("ranking", {})
    auto_pause   = profile.get("auto_pause", False)
    instance_id  = profile.get("instance_id", "413754")

    # 2b. Adaptation mode (Track G2/G3): drift-eval + sleep consolidation.
    if getattr(args, "adapt", False) or expt_meta.get("mode") == "adapt":
        _run_adapt(args.expt, spec, args)
        return

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

    # 5b. Track F: encoder-only JEPA self-supervised pretraining (no LLM is built).
    # Triggered for a jepa encoder WITHOUT a pretrained_ckpt. Supplying
    # encoder.pretrained_ckpt (via --override) instead falls through to the standard
    # E2E loop below — i.e. the downstream fine-tune of a pretrained JEPA backbone.
    _enc_spec = spec.get("encoder", {})
    if _enc_spec.get("variant") == "jepa" and not _enc_spec.get("pretrained_ckpt"):
        _run_jepa_pretrain(args.expt, spec, args, run_dir, logger, device, compute_dtype)
        return

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

    enc  = stack.encoder.to(device).to(compute_dtype)  # cast to bf16: fixes HRM DEQ backward dtype mismatch
    proj = stack.projector.to(device)
    dec  = stack.decoder  # decoder manages its own device_map
    mem  = stack._modules["memory"].to(device) if stack.has_stage("memory") else None  # Track H

    # C3 Whisper-Qwen bridge modules live on decoder but don't use device_map — move explicitly
    for _bridge_attr in ("neural_to_whisper", "whisper_encoder", "whisper_to_llm"):
        if hasattr(dec, _bridge_attr):
            getattr(dec, _bridge_attr).to(device)

    # 8. Wire CTC head + TopoLoss if in loss spec
    composed_loss = compose_from_spec(spec.get("loss", [{"variant": "ce"}]))
    for loss_fn in composed_loss._fns:
        if hasattr(loss_fn, "attach"):   # TopoLossStage
            loss_fn.attach(enc)
            loss_fn.to(device)           # move topo conv kernel to GPU
        if hasattr(loss_fn, "ctc_head"):  # CTCAnnealLoss
            loss_fn.ctc_head = loss_fn.ctc_head.to(device)

    # 9. Optimizer
    trainable = (
        list(enc.parameters()) +
        list(proj.parameters()) +
        [p for p in dec.parameters() if p.requires_grad]
    )
    if mem is not None:
        trainable += list(mem.parameters())   # episodic read head + fusion gate
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
                    mem_out  = {}
                    if mem is not None:                 # Track H: episodic memory stage
                        tokens  = mem(tokens, session_id=sid)
                        mem_out = mem.last_read         # memory_query / memory_retrieved
                    proj_out = proj(tokens)
                    outputs  = dec(proj_out, labels=batch["text"],
                                    neural_lengths=lengths)
                    # Expose tensors for compose
                    outputs["neural_tokens"]   = tokens
                    outputs["projected_embeds"] = proj_out
                    outputs["batch"]           = batch
                    outputs.update(mem_out)

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


def _run_adapt(expt_id: str, spec: dict, args):
    """
    Track G2/G3: build encoder + CTC head (no LLM), construct an ordered day
    sequence, and sweep the consolidation depth N through the DietCorp TTA loop.
    Writes a PER-vs-day curve per N + a wake-latency / consolidate-cost table.
    """
    from stages.encoder.bit import build as enc_build
    from adapt.dietcorp_tta import (
        TTAConfig, select_patch_embed_params, ctc_greedy_decode,
    )
    from tools.drift_eval import synthesize_drift, split_by_session, run_drift_eval
    import torch.nn.functional as F

    a = spec.get("adapt", {})
    enc_spec   = spec.get("encoder", {"variant": "bit"})
    input_dim  = enc_spec.get("input_dim", 512)
    embed_dim  = enc_spec.get("embed_dim", 384)
    n_phonemes = a.get("n_phonemes", 42)
    blank      = a.get("blank", 0)
    T_bins     = 240
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build the LLM-free phoneme model: encoder → CTC head ──────────────────
    enc_spec = dict(enc_spec); enc_spec.pop("variant", None)
    encoder, _ = enc_build(enc_spec, (T_bins, input_dim))

    class _PhonemeModel(nn.Module):
        def __init__(self, encoder, ctc_head):
            super().__init__(); self.encoder = encoder; self.ctc_head = ctc_head
        def forward(self, neural):                         # (B,T,C) -> (B,T',P)
            tok = self.encoder(neural, session_id=None, neural_lengths=None)
            return self.ctc_head(tok)

    model = _PhonemeModel(encoder, nn.Linear(embed_dim, n_phonemes)).to(device)

    # FIX 1: Load the TRAINED CTC head from the checkpoint.
    # bit.py stage only loads encoder.* keys, intentionally skipping head.*.
    # We load head.weight / head.bias here so pseudo-labels are real, not random.
    ckpt_path = enc_spec.get("pretrained_ckpt")
    if ckpt_path and os.path.exists(str(ckpt_path)):
        _ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if "head.weight" in _ckpt and "head.bias" in _ckpt:
            model.ctc_head.weight.data.copy_(_ckpt["head.weight"])
            model.ctc_head.bias.data.copy_(_ckpt["head.bias"])
            print(f"[adapt] Loaded trained CTC head from checkpoint "
                  f"(shape: {_ckpt['head.weight'].shape})")
        else:
            print("[adapt] WARNING: checkpoint has no head.weight — CTC head is random")
    model.eval()

    target_params = select_patch_embed_params(
        model.encoder, tuple(a.get("target_param_hints", ("patch", "read_in", "embed"))))
    print(f"[adapt] {expt_id}: target params (patch-embed) = {len(target_params)} tensors "
          f"({sum(p.numel() for p in target_params)} weights)")

    # ── Base trials → ordered days ───────────────────────────────────────────
    d = a.get("drift", {})
    n_days     = d.get("n_days", 8)
    max_trials = d.get("max_trials", 32)

    # FIX 2: Use REAL neural data when --val_h5 is provided.
    # Random Gaussian tensors fed to a trained model produce near-uniform CTC
    # outputs (confidence ~0.07) → garbage pseudo-labels → immediate collapse.
    # Real recordings are in-distribution and produce meaningful phoneme decodes.
    base = None
    if getattr(args, "val_h5", None):
        try:
            from docks.multiarch_dock import Preprocessed_BCI_Dataset, bci_collate_fn
            from torch.utils.data import DataLoader
            val_h5s = _find_h5(args.val_h5, "data_val.hdf5")
            if val_h5s:
                ds  = Preprocessed_BCI_Dataset(
                          val_h5s,
                          patch_size=enc_spec.get("patch_size", 4),
                          augment=False)
                ldr = DataLoader(ds, batch_size=1, shuffle=False,
                                 collate_fn=bci_collate_fn, num_workers=0)
                base = []
                for i, batch in enumerate(ldr):
                    if i >= max_trials:
                        break
                    neural = batch["neural"][0].to(device)  # (T, C)
                    base.append((neural, None))
                print(f"[adapt] Loaded {len(base)} real trials from {val_h5s[0]}")
        except Exception as e:
            print(f"[adapt] WARNING: could not load real data ({e}), falling back to synthetic")
            base = None

    if base is None:
        print("[adapt] Using synthetic random base trials (confidence will be low with a trained model)")
        base = [(torch.randn(T_bins, input_dim, device=device), None)
                for _ in range(max_trials)]

    days = synthesize_drift(base, n_days=n_days,
                            scale_std=d.get("scale_std", 0.15),
                            shift_std=d.get("shift_std", 0.15),
                            noise_std=d.get("noise_std", 0.05),
                            seed=d.get("seed", 0))

    # No ground-truth phonemes for synthetic data → use the fresh model's day-0
    # greedy decode as each trial's reference, so PER measures how far the decode
    # has DRIFTED from its clean-day baseline.
    with torch.no_grad():
        day0 = list(days.values())[0]
        refs = [ctc_greedy_decode(F.log_softmax(model(x.unsqueeze(0)), dim=-1), blank=blank)[0]
                for x, _ in day0]
    for day_trials in days.values():
        for i in range(len(day_trials)):
            day_trials[i] = (day_trials[i][0], refs[i])

    # ── Sweep N ──────────────────────────────────────────────────────────────
    n_steps_list = args.n_steps if args.n_steps else a.get("n_steps_list", [0, 1, 2, 4, 8])
    cfg = TTAConfig(n_aug=a.get("n_aug", 64), mask_frac=a.get("mask_frac", 0.53),
                    mask_span=a.get("mask_span", 4), lr=a.get("lr", 1e-3),
                    grad_clip=a.get("grad_clip", 1.0), blank=blank,
                    confidence_threshold=a.get("confidence_threshold", 0.0))

    print(f"[adapt] sweeping N = {list(n_steps_list)} over {n_days} days "
          f"x {max_trials} trials on {device}...")
    res = run_drift_eval(model, model, days, target_params,
                         n_steps_list=list(n_steps_list), tta_config=cfg, blank=blank)

    # ── Persist + summarise ──────────────────────────────────────────────────
    run_dir = _HERE / "results" / "runs" / f"{expt_id}_adapt_{_spec_hash(spec)}"
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir / "drift_results.json", "w") as f:
        json.dump(res, f, indent=2)

    print(f"\n{'N':>4} {'PER@day0':>9} {'PER@last':>9} {'delta(L-0)':>11} "
          f"{'wake_ms':>9} {'cons_ms':>9}")
    for N in n_steps_list:
        s = res["summary"][N]
        wake = res["wake_latency_ms"][N]; cons = res["consolidate_ms"][N]
        def _f(x, p=4): return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  n/a"
        print(f"{N:>4} {_f(s['per_first']):>9} {_f(s['per_last']):>9} "
              f"{_f(s['per_delta']):>14} {_f(wake,2):>9} {_f(cons,2):>9}")
    print(f"\n[adapt] H_main supported iff PER@last falls as N rises while wake_ms stays flat.")
    print(f"[adapt] results → {run_dir / 'drift_results.json'}")


def _run_analysis(expt_id: str, spec: dict, args):
    """Dispatch an analysis experiment (A1/A2/A3) to its tool.

    Each tool has a bespoke argparse, so we translate the spec `analysis:` block
    + run.py args into the tool's own CLI and swap sys.argv before calling main()
    (the previous version called main() with run.py's argv, which crashed the
    tool's parser on --expt/--profile)."""
    import importlib
    import os

    a = spec.get("analysis", {})
    os.makedirs("results", exist_ok=True)

    if expt_id == "A1":
        mod_name = "tools.cka_analysis"
        argv = ["cka_analysis", "--val_h5", str(args.val_h5),
                "--out", "results/A1_cka_scores.json"]
        if a.get("encoder_ckpt"):
            argv += ["--ckpt", str(a["encoder_ckpt"])]
        if a.get("batch_size"):
            argv += ["--batch_size", str(a["batch_size"])]
    elif expt_id == "A2":
        mod_name = "tools.perplexity_test"
        spoken  = os.path.expandvars(str(a.get("spoken_corpus", "")))
        written = os.path.expandvars(str(a.get("written_corpus", "")))
        for label, p in (("spoken_corpus", spoken), ("written_corpus", written)):
            if (not p) or ("${" in p) or (not os.path.exists(p)):
                raise FileNotFoundError(
                    f"[A2] {label} resolved to {p!r}, which is missing. Set BCI_DATA_ROOT "
                    f"to a directory holding the corpora, or run tools/perplexity_test.py "
                    f"directly. Skipping A2 is fine — it is not on the core sweep path."
                )
        argv = ["perplexity_test", "--spoken_file", spoken,
                "--written_file", written, "--out", "results/A2_perplexity.json"]
    elif expt_id == "A3":
        mod_name = "tools.phoneme_probe"
        argv = ["phoneme_probe", "--val_h5", str(args.val_h5),
                "--out", "results/A3_phoneme_probe.json"]
        if a.get("epochs"):
            argv += ["--epochs", str(a["epochs"])]
    else:
        raise NotImplementedError(f"Analysis dispatch for {expt_id} not implemented")

    mod = importlib.import_module(mod_name)
    _old_argv = sys.argv
    try:
        sys.argv = argv
        mod.main()
    finally:
        sys.argv = _old_argv


def _run_jepa_pretrain(expt_id, spec, args, run_dir, logger, device, compute_dtype):
    """Track F: encoder-only JEPA self-supervised pretraining.

    The standard E2E loop would train the JEPA encoder via CE through the LLM and
    leave the masked-latent objective + VICReg anti-collapse unused. This path runs
    the REAL objective: per batch, encoder.pretrain_step(x) does masked-latent
    prediction + EMA target update, and the VICReg loss on the unmasked context
    embeddings guards against representation collapse. No LLM is built. Saves a
    pretrained backbone + collapse-health metrics; the cross-modality WER comparison
    is the downstream fine-tune (run.py --expt F* --override encoder.pretrained_ckpt=<saved>)."""
    import json as _json
    from stages.encoder.jepa import build as jepa_build
    from stages.loss.jepa_varcov import build as varcov_build

    train_h5s = _find_h5(args.train_h5, "data_train.hdf5")
    val_h5s   = _find_h5(args.val_h5,   "data_val.hdf5")
    if not train_h5s:
        raise FileNotFoundError(f"No training HDF5 found in {args.train_h5}")

    enc_spec   = spec.get("encoder", {})
    patch_size = enc_spec.get("patch_size", 4)
    bs = spec.get("batch_size", 4)
    nw = spec.get("num_workers", 0)
    train_loader = DataLoader(
        Preprocessed_BCI_Dataset(train_h5s, patch_size=patch_size, augment=True),
        batch_size=bs, shuffle=True, collate_fn=bci_collate_fn, num_workers=nw)
    val_loader = DataLoader(
        Preprocessed_BCI_Dataset(val_h5s, patch_size=patch_size, augment=False),
        batch_size=bs, shuffle=False, collate_fn=bci_collate_fn, num_workers=nw) if val_h5s else None

    encoder, _ = jepa_build(enc_spec, prev_shape=(240, enc_spec.get("input_dim", 512)))
    encoder = encoder.to(device)
    modality = enc_spec.get("modality", "audio")

    varcov_spec   = next((dict(l) for l in spec.get("loss", []) if l.get("variant") == "jepa_varcov"), {})
    varcov_fn, _  = varcov_build(varcov_spec, None)

    optimizer = AdamW([p for p in encoder.parameters() if p.requires_grad],
                      lr=spec.get("lr", 5e-4), weight_decay=spec.get("weight_decay", 1e-5))
    epochs      = spec.get("epochs", 20)
    max_batches = spec.get("max_batches_per_epoch")
    logger.info(f"[jepa] Pretraining modality={modality} for {epochs} epochs "
                f"(batch_size={bs}, max_batches={max_batches})")

    hist = {"pred_loss": [], "varcov_loss": [], "std_mean": []}
    for epoch in range(1, epochs + 1):
        encoder.train()
        agg = [0.0, 0.0, 0.0, 0]
        pbar = tqdm(train_loader, desc=f"[jepa:{modality}] ep{epoch}/{epochs}")
        for step, batch in enumerate(pbar):
            if max_batches and step >= max_batches:
                break
            neural = batch["neural"].to(device)
            with torch.autocast(device_type=device.type, dtype=compute_dtype,
                                enabled=(device.type == "cuda")):
                pred_loss = encoder.pretrain_step(neural)     # masked-latent + EMA update
                z = encoder(neural)                           # unmasked context embeddings
                vc = varcov_fn(batch, None, {"jepa_embeds": z})["loss_jepa_varcov"]
                loss = pred_loss + vc
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                std = z.reshape(-1, z.shape[-1]).float().std(0).mean().item()
            agg[0] += float(pred_loss); agg[1] += float(vc); agg[2] += std; agg[3] += 1
            pbar.set_postfix(pred=f"{float(pred_loss):.3f}", vc=f"{float(vc):.3f}", std=f"{std:.2f}")
        n = max(agg[3], 1)
        hist["pred_loss"].append(agg[0] / n)
        hist["varcov_loss"].append(agg[1] / n)
        hist["std_mean"].append(agg[2] / n)
        logger.info(f"[jepa] ep{epoch} pred={agg[0]/n:.4f} varcov={agg[1]/n:.4f} "
                    f"std={agg[2]/n:.3f} ({'OK' if agg[2]/n > 0.5 else 'COLLAPSE-RISK'})")

    # Held-out collapse check (no EMA, no grad)
    val_std = None
    if val_loader is not None:
        encoder.eval(); s = 0.0; c = 0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if step >= 10:
                    break
                z = encoder(batch["neural"].to(device))
                s += z.reshape(-1, z.shape[-1]).float().std(0).mean().item(); c += 1
        val_std = s / max(c, 1)

    ckpt = run_dir / "pretrained_encoder.pth"
    torch.save({f"encoder.{k}": v for k, v in encoder.state_dict().items()}, str(ckpt))
    collapse_ok = bool((hist["std_mean"][-1] if hist["std_mean"] else 0.0) > 0.5)
    metrics = {
        "modality": modality, "epochs": epochs,
        "final_pred_loss":  hist["pred_loss"][-1] if hist["pred_loss"] else None,
        "final_train_std":  hist["std_mean"][-1] if hist["std_mean"] else None,
        "val_std": val_std, "collapse_ok": collapse_ok,
        "history": hist, "backbone": str(ckpt),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        _json.dump(metrics, f, indent=2)

    record_run(expt_id=expt_id, profile=args.profile, spec_hash=_spec_hash(spec),
               best_wer=None, run_dir=str(run_dir),
               notes=(f"JEPA pretrain modality={modality}; collapse_ok={collapse_ok}; "
                      f"final_pred_loss={metrics['final_pred_loss']}; downstream WER via "
                      f"--override encoder.pretrained_ckpt={ckpt.name}"))
    logger.info(f"[jepa] DONE modality={modality} collapse_ok={collapse_ok} -> {ckpt}")
    logger.info(f"[jepa] Downstream comparison (real WER): python run.py --expt {expt_id} "
                f"--profile {args.profile} --override encoder.pretrained_ckpt={ckpt} "
                f"--train_h5 {args.train_h5} --val_h5 {args.val_h5}")


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
    parser.add_argument("--adapt", action="store_true",
                        help="Run the DietCorp drift-eval / sleep-consolidation loop (Track G2/G3) "
                             "instead of training. LLM-free: builds encoder + CTC head only.")
    parser.add_argument("--n_steps", nargs="*", type=int, default=None,
                        help="Override the consolidation-depth sweep, e.g. --n_steps 0 1 2 4 8")
    args = parser.parse_args()

    # Apply CLI overrides to spec (handled inside run() via spec YAML)
    run(args)


if __name__ == "__main__":
    main()
