"""
tools/diff_specs.py
-------------------
Controlled-A/B enforcer.

Two experiment specs that claim to differ in only one axis (e.g. JEPA
audio vs video) must differ in EXACTLY the lines you expect — nothing else.
A silent extra difference (a stray param-count bump, a different patch_size)
is how an "audio beats video" result becomes a confound instead of science.

Usage (CLI):
    python tools/diff_specs.py specs/F1_audio_jepa.yaml specs/F2_video_jepa.yaml \\
        --expect-keys modality

Returns exit 0 if the only differing top-level *keys* are the expected ones,
else exit 1 with a diff. Also importable: assert_differs_only(a, b, keys).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict to dotted keys, e.g. {'encoder.modality': 'audio'}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(_flatten(item, f"{key}[{i}]."))
                else:
                    out[f"{key}[{i}]"] = item
        else:
            out[key] = v
    return out


def diff_keys(a: dict, b: dict) -> dict:
    """Return {dotted_key: (a_val, b_val)} for every key whose value differs."""
    fa, fb = _flatten(a), _flatten(b)
    diffs = {}
    for k in set(fa) | set(fb):
        va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
        if va != vb:
            diffs[k] = (va, vb)
    return diffs


def assert_differs_only(spec_a: dict, spec_b: dict, expect_keys: list[str]) -> None:
    """Raise AssertionError unless the differing keys' leaf-names ⊆ expect_keys."""
    diffs = diff_keys(spec_a, spec_b)
    unexpected = {
        k: v for k, v in diffs.items()
        if k.split(".")[-1] not in set(expect_keys)
    }
    if unexpected:
        lines = [f"  {k}: {va!r} != {vb!r}" for k, (va, vb) in unexpected.items()]
        raise AssertionError(
            "Specs differ in UNEXPECTED keys (confound risk):\n"
            + "\n".join(lines)
            + f"\nOnly these keys were allowed to differ: {expect_keys}"
        )


def main():
    ap = argparse.ArgumentParser(description="Assert two specs differ in only expected keys")
    ap.add_argument("spec_a")
    ap.add_argument("spec_b")
    ap.add_argument("--expect-keys", nargs="+", default=["modality"],
                    help="Leaf key names allowed to differ (default: modality)")
    args = ap.parse_args()

    a = yaml.safe_load(Path(args.spec_a).read_text())
    b = yaml.safe_load(Path(args.spec_b).read_text())

    try:
        assert_differs_only(a, b, args.expect_keys)
    except AssertionError as e:
        print(f"[diff_specs] FAIL\n{e}", file=sys.stderr)
        sys.exit(1)

    diffs = diff_keys(a, b)
    print(f"[diff_specs] OK — {args.spec_a} vs {args.spec_b} "
          f"differ only in {sorted(diffs.keys())}")
    sys.exit(0)


if __name__ == "__main__":
    main()
