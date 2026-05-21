from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


TARGET_NAMES = ("thumb", "index", "middle", "ring", "little")


@dataclass(frozen=True)
class PrepConfig:
    subject: int = 1
    sfreq: float = 200.0
    window_sec: float = 1.5
    stride_sec: float = 0.25
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    data_dir: str = "data"
    braindecode_cache: str = "data/braindecode_cache"


def _load_raw_from_braindecode(config: PrepConfig):
    try:
        from braindecode.datasets import BCICompetitionIVDataset4
    except Exception as exc:  # pragma: no cover - dependency/runtime guard
        raise RuntimeError(
            "Braindecode is required for real ECoG data. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    try:
        dataset = BCICompetitionIVDataset4(
            subject_ids=[config.subject],
            data_folder=config.braindecode_cache,
        )
    except TypeError:
        dataset = BCICompetitionIVDataset4(subject_ids=[config.subject])

    raws = []
    for item in getattr(dataset, "datasets", []):
        raw = getattr(item, "raw", None)
        if raw is not None:
            raws.append(raw.copy())

    if not raws:
        raise RuntimeError("Could not find MNE Raw objects in the Braindecode dataset.")

    if len(raws) == 1:
        return raws[0]

    import mne

    return mne.concatenate_raws(raws)


def _pick_channels(raw):
    names = list(raw.ch_names)
    lower = [name.lower() for name in names]
    types = raw.get_channel_types()

    misc_indices = [idx for idx, kind in enumerate(types) if kind == "misc"]
    ecog_indices = [idx for idx, kind in enumerate(types) if kind == "ecog"]

    target_indices = misc_indices[: len(TARGET_NAMES)]
    if len(target_indices) != len(TARGET_NAMES):
        target_indices = []
        for target in TARGET_NAMES:
            matches = [idx for idx, name in enumerate(lower) if target in name]
            if matches:
                target_indices.append(matches[0])

    if len(target_indices) != len(TARGET_NAMES):
        # The BCI IV-4 files commonly store the glove trajectories as the last five channels.
        target_indices = list(range(len(names) - len(TARGET_NAMES), len(names)))

    if not ecog_indices:
        ecog_indices = [idx for idx in range(len(names)) if idx not in set(target_indices)]
    if len(ecog_indices) < 4:
        raise RuntimeError(
            "Could not identify enough ECoG channels. Inspect raw.ch_names and update "
            "_pick_channels for this dataset layout."
        )
    return ecog_indices, target_indices


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.maximum(std, eps)


def _make_windows(
    ecog: np.ndarray,
    targets: np.ndarray,
    sfreq: float,
    window_sec: float,
    stride_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    window = int(round(window_sec * sfreq))
    stride = int(round(stride_sec * sfreq))
    if window <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must produce positive sample counts.")
    if ecog.shape[1] < window:
        raise ValueError("Recording is shorter than one window.")

    starts = np.arange(0, ecog.shape[1] - window + 1, stride, dtype=np.int64)
    x = np.empty((len(starts), ecog.shape[0], window), dtype=np.float32)
    y = np.empty((len(starts), targets.shape[0]), dtype=np.float32)

    for out_idx, start in enumerate(starts):
        stop = start + window
        x[out_idx] = ecog[:, start:stop]
        # Predict the local trajectory state near the end of the neural context window.
        tail_start = max(start, stop - max(1, window // 5))
        y[out_idx] = targets[:, tail_start:stop].mean(axis=1)

    return x, y


def _split_time_ordered(
    x: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, np.ndarray]:
    n = len(x)
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough windows for train/val/test split.")

    return {
        "x_train": x[:n_train],
        "y_train": y[:n_train],
        "x_val": x[n_train : n_train + n_val],
        "y_val": y[n_train : n_train + n_val],
        "x_test": x[n_train + n_val :],
        "y_test": y[n_train + n_val :],
    }


def prepare(config: PrepConfig) -> Path:
    out_dir = Path(config.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ecog_fingerflex_subject{config.subject}.npz"

    raw = _load_raw_from_braindecode(config)
    raw.load_data()
    raw.resample(config.sfreq)
    high_cut = min(100.0, config.sfreq / 2.0 - 1.0)
    if high_cut > 1.0:
        raw.filter(l_freq=1.0, h_freq=high_cut, picks="ecog")

    ecog_indices, target_indices = _pick_channels(raw)
    data = raw.get_data().astype(np.float32)
    ecog = _zscore(data[ecog_indices] / 1e-6)
    targets = _zscore(data[target_indices])

    x, y = _make_windows(
        ecog=ecog,
        targets=targets,
        sfreq=float(raw.info["sfreq"]),
        window_sec=config.window_sec,
        stride_sec=config.stride_sec,
    )
    splits = _split_time_ordered(x, y, config.val_fraction, config.test_fraction)

    metadata = {
        **asdict(config),
        "sfreq_actual": float(raw.info["sfreq"]),
        "ecog_channels": [raw.ch_names[idx] for idx in ecog_indices],
        "target_channels": [raw.ch_names[idx] for idx in target_indices],
        "n_windows": int(len(x)),
        "n_ecog_channels": int(x.shape[1]),
        "window_samples": int(x.shape[2]),
    }

    np.savez_compressed(out_path, **splits, metadata=json.dumps(metadata, indent=2))
    print(json.dumps({"prepared": str(out_path), **metadata}, indent=2))
    return out_path


def parse_args() -> PrepConfig:
    parser = argparse.ArgumentParser(description="Prepare BCI IV-4 ECoG finger-flexion windows.")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--window-sec", type=float, default=1.5)
    parser.add_argument("--stride-sec", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--braindecode-cache", default="data/braindecode_cache")
    return PrepConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    prepare(parse_args())
