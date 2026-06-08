#!/usr/bin/env python3
"""
autoresearch/preflight.py

Environment probe + registry runnability classifier.
Run this once before starting a sweep.  Writes autoresearch/runnable.json,
which the sweep harness reads to know which experiments to queue.

Usage (Windows):
    $env:PYTHONIOENCODING="utf-8"
    py -3 autoresearch/preflight.py
    py -3 autoresearch/preflight.py --registry registry.yaml --skip-shape-gate
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# L0 — environment probe
# ---------------------------------------------------------------------------

def probe_env() -> dict:
    """
    Probe torch/CUDA availability and one 4-bit bnb forward.
    Returns a flat dict; all values are JSON-serialisable.
    """
    result: dict = {}

    # torch + CUDA
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            result["cuda_device"]   = props.name
            result["vram_total_mb"] = round(props.total_memory / 1024**2)
            result["cuda_version"]  = torch.version.cuda
    except ImportError as exc:
        result["torch_error"] = str(exc)
        return result

    # bitsandbytes 4-bit NF4 forward (the gating check for all E2E runs)
    try:
        import torch
        import bitsandbytes as bnb
        layer = bnb.nn.Linear4bit(
            32, 32,
            bias=False,
            quant_type="nf4",
            compute_dtype=torch.bfloat16,
        )
        # move to CUDA only if available
        x = torch.randn(2, 32)
        if torch.cuda.is_available():
            layer = layer.cuda()
            x = x.cuda()
        _ = layer(x)
        result["bnb_4bit"] = "ok"
    except Exception as exc:
        result["bnb_4bit"] = f"FAIL: {exc}"

    # mamba-ssm (check only — not a hard requirement, B3_mamba is deferred if absent)
    try:
        import mamba_ssm  # noqa: F401
        result["mamba_ssm"] = "available"
    except ImportError:
        result["mamba_ssm"] = "unavailable"

    # transformers / peft
    for pkg in ("transformers", "peft", "h5py", "yaml"):
        import importlib
        spec = importlib.util.find_spec(pkg)
        result[f"pkg_{pkg}"] = "ok" if spec else "MISSING"

    return result


# ---------------------------------------------------------------------------
# Shape gate — delegated entirely to existing pytest suite
# ---------------------------------------------------------------------------

def run_shape_gate(skip: bool = False) -> dict:
    if skip:
        return {"passed": None, "skipped": True, "output": "skipped by --skip-shape-gate"}

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    res = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_stage_shapes.py",
         "-v", "--tb=short", "-q", "--no-header"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
        env=env,
    )
    output = (res.stdout or "") + (res.stderr or "")
    return {
        "passed":    res.returncode == 0,
        "exit_code": res.returncode,
        "output":    output[-3000:],   # last 3 KB to avoid log bloat
    }


# ---------------------------------------------------------------------------
# Registry classification
# ---------------------------------------------------------------------------

_SCOPE_TRACKS = {"A", "B", "C", "D", "E", "F"}

# Experiments that require mamba-ssm (won't build on Windows)
_MAMBA_REQUIRED = {"B3_mamba"}

# Names/keys that signal a 7B+ model in the LLM decoder
_LARGE_MODEL_SIGNALS = {"7b", "5.6b", "phi-4", "audio-7b"}


def classify(exp_id: str, exp: dict, env: dict) -> tuple[bool, list[str], str]:
    """
    Returns (runnable_local, blockers, notes).
    blockers is empty iff runnable_local is True.
    """
    state = exp.get("state", "")
    if state == "skeleton":
        # General guard: any experiment still marked `state: skeleton` in the registry is not yet
        # implemented (its stage is stubbed) — defer until the stage is real. (Track F is no
        # longer skeleton — its JEPA backbones are live; this branch guards future skeletons.)
        return False, ["state=skeleton — stage not yet implemented; defer until real"], ""

    local_ok    = exp.get("local_ok", False)
    cloud_req   = exp.get("cloud_required", False)

    if cloud_req or not local_ok:
        return False, ["cloud_required or local_ok=false in registry"], ""

    blockers: list[str] = []
    notes:    list[str] = []

    # B3_mamba needs mamba-ssm (fails to build on Windows)
    if exp_id in _MAMBA_REQUIRED and env.get("mamba_ssm") != "available":
        blockers.append("mamba-ssm unavailable — run B3 (GRU fallback) locally")
        notes.append("defer B3_mamba to cloud/Linux")

    # 7B+ models — inferred from experiment name
    name_lower = exp.get("name", "").lower()
    if any(sig in name_lower for sig in _LARGE_MODEL_SIGNALS):
        blockers.append("7B/5.6B decoder exceeds 6 GB VRAM — cloud only")

    # bitsandbytes 4-bit: required for all training experiments with an LLM decoder
    if exp.get("train_required", False) and exp.get("track") in _SCOPE_TRACKS:
        bnb_ok = env.get("bnb_4bit", "").startswith("ok")
        if not bnb_ok:
            blockers.append("bitsandbytes 4-bit FAIL — E2E training needs it")

    # B4 MoE: assert aux_loss_weight > 0 in spec
    if exp_id == "B4":
        spec_path = REPO_ROOT / exp.get("spec_ref", "")
        if spec_path.exists():
            import yaml
            with open(spec_path) as f:
                spec_data = yaml.safe_load(f)
            aux = spec_data.get("aux_loss_weight", 0)
            if not (isinstance(aux, (int, float)) and aux > 0):
                blockers.append(f"B4 aux_loss_weight={aux!r} — must be >0 to prevent expert collapse")
        else:
            notes.append("B4 spec not found — cannot verify aux_loss_weight")

    return len(blockers) == 0, blockers, "; ".join(notes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch preflight check")
    parser.add_argument("--registry",         default=str(REPO_ROOT / "registry.yaml"))
    parser.add_argument("--skip-shape-gate",  action="store_true",
                        help="Skip pytest test_stage_shapes.py (saves ~10 s on re-runs)")
    args = parser.parse_args()

    print("=" * 60)
    print("  autoresearch/preflight.py")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()

    # L0 — env probe
    print("L0  Environment probe")
    print("    ─────────────────────────────────────────────")
    env = probe_env()
    print(f"    torch:       {env.get('torch_version', 'MISSING')}")
    print(f"    CUDA:        {env.get('cuda_available')}  "
          f"device={env.get('cuda_device', '?')}  "
          f"vram={env.get('vram_total_mb', '?')} MB  "
          f"cuda={env.get('cuda_version', '?')}")
    print(f"    bnb 4-bit:   {env.get('bnb_4bit', '?')}")
    print(f"    mamba-ssm:   {env.get('mamba_ssm', '?')}")
    for pkg in ("transformers", "peft", "h5py", "yaml"):
        v = env.get(f"pkg_{pkg}", "?")
        print(f"    {pkg:15s}  {v}")

    bnb_ok = env.get("bnb_4bit", "").startswith("ok")
    mamba_ok = env.get("mamba_ssm") == "available"

    if not bnb_ok:
        print()
        print("    !! bitsandbytes 4-bit FAILED.")
        print("       All E2E training experiments will be deferred to cloud.")
        print("       To fix: pin a Windows-compatible wheel, e.g.:")
        print("         pip install bitsandbytes==0.44.1")
        print("       Or run the sweep on A100 (see autoresearch/A100_RUNBOOK.md).")

    # L1 — shape gate
    print()
    print("L1  Shape gate  (tests/test_stage_shapes.py)")
    print("    ─────────────────────────────────────────────")
    shape = run_shape_gate(skip=args.skip_shape_gate)
    if shape.get("skipped"):
        print("    SKIPPED")
    elif shape["passed"]:
        print("    PASS")
    else:
        print(f"    FAIL  (exit {shape['exit_code']})")
        print()
        print(shape["output"])
        print("    !! Fix test_stage_shapes.py failures before running any experiment.")

    # L2 — registry classification
    print()
    print("L2  Registry classification  (registry.yaml)")
    print("    ─────────────────────────────────────────────")

    import yaml
    with open(args.registry, encoding="utf-8") as f:
        reg = yaml.safe_load(f)

    experiments = reg.get("experiments", {})
    runnable: list[dict] = []
    deferred: list[dict] = []
    skipped:  list[dict] = []

    for exp_id, exp in experiments.items():
        track = exp.get("track", "?")

        if track not in _SCOPE_TRACKS:
            skipped.append({"expt": exp_id, "track": track, "reason": "out of A/B/D/E scope"})
            continue

        ok, blockers, notes = classify(exp_id, exp, env)
        row = {
            "expt":                exp_id,
            "track":               track,
            "name":                exp.get("name", ""),
            "runnable_local":      ok,
            "blockers":            blockers,
            "depends_on":          exp.get("depends_on", []),
            "expected_runtime_min": exp.get("expected_runtime_min"),
            "expected_wer_band":   exp.get("expected_wer_band"),
            "train_required":      exp.get("train_required", True),
            "spec_ref":            exp.get("spec_ref", ""),
            "notes":               notes,
        }
        (runnable if ok else deferred).append(row)

    # Print runnable
    print(f"\n    RUNNABLE LOCALLY ({len(runnable)} experiments)")
    print(f"    {'ID':<15}  {'Track':<6}  {'Est min':>8}  Depends on")
    print(f"    {'-'*15}  {'-'*6}  {'-'*8}  {'-'*30}")
    for r in runnable:
        deps = ", ".join(r["depends_on"]) if r["depends_on"] else "—"
        rt   = str(r["expected_runtime_min"]) if r["expected_runtime_min"] else "?"
        print(f"    {r['expt']:<15}  [{r['track']}]     {rt:>6} m  {deps}")

    # Print deferred
    print(f"\n    DEFERRED / CLOUD ({len(deferred)} experiments)")
    for d in deferred:
        blocker = d["blockers"][0] if d["blockers"] else "unknown"
        print(f"    {d['expt']:<15}  [{d['track']}]  {blocker}")

    # Print out-of-scope
    if skipped:
        print(f"\n    OUT OF SCOPE ({len(skipped)}): "
              + ", ".join(s["expt"] for s in skipped))

    # Total time estimate for runnable training experiments
    total_min = sum(
        r["expected_runtime_min"] or 0
        for r in runnable
        if r["train_required"]
    )
    analysis_min = sum(
        r["expected_runtime_min"] or 0
        for r in runnable
        if not r["train_required"]
    )
    print()
    print(f"    Estimated sweep time (local, sequential):")
    print(f"      Training experiments:  ~{total_min} min")
    print(f"      Analysis experiments:  ~{analysis_min} min")
    print(f"      Total:                 ~{total_min + analysis_min} min "
          f"(~{(total_min + analysis_min)//60} h {(total_min + analysis_min)%60} m)")
    print()
    if not mamba_ok:
        print("    Note: B3_mamba deferred because mamba-ssm is unavailable.")
        print("          On A100/Linux: pip install mamba-ssm causal-conv1d  (~3 min)")
        print("          See autoresearch/A100_RUNBOOK.md for full cloud path.")

    # Write output
    out_path = REPO_ROOT / "autoresearch" / "runnable.json"
    output = {
        "generated":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env":         env,
        "shape_gate":  shape,
        "runnable":    runnable,
        "deferred":    deferred,
        "skipped":     skipped,
        "estimated_local_sweep_min": total_min + analysis_min,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"    Written: {out_path}")
    print()

    if shape.get("passed") is False:
        print("WARNING: shape gate failed — fix before running the sweep.")
        sys.exit(1)

    if not bnb_ok:
        print("WARNING: bitsandbytes 4-bit unavailable — sweep will run analysis-only locally.")
        print("         Full sweep → A100 (see A100_RUNBOOK.md).")

    print("Preflight complete.")
    print("Next: py -3 autoresearch/sweep.py  (or review runnable.json first)")


if __name__ == "__main__":
    main()
