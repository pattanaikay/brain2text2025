from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    model: str = "cnn"
    data_path: str = "data/ecog_fingerflex_subject1.npz"
    run_dir: str = "runs/dev"
    seed: int = 13
    subject: int = 1
    budget_minutes: float = 3.0
    batch_size: int = 0
    lr: float = 3e-4
    weight_decay: float = 1e-3
    max_epochs: int = 50
    device: str = "auto"
    num_workers: int = 0
    use_amp: bool = True


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def pearson_per_target(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred = pred - pred.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    denom = np.sqrt((pred**2).sum(axis=0) * (target**2).sum(axis=0))
    return ((pred * target).sum(axis=0) / np.maximum(denom, 1e-12)).astype(np.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LightweightCNN(nn.Module):
    def __init__(self, channels: int, samples: int, targets: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=7, stride=2, padding=3, groups=4, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2, groups=4, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, targets),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pred": self.net(x)}


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # x: batch, heads, time, head_dim
    head_dim = x.shape[-1]
    half = head_dim // 2
    freqs = torch.arange(0, half, 1, device=x.device, dtype=x.dtype)
    inv_freq = 1.0 / (10000 ** (freqs / max(1, half)))
    positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
    angles = torch.einsum("t,d->td", positions, inv_freq)
    angles = torch.repeat_interleave(angles, 2, dim=-1)[None, None, :, :head_dim]
    return (x * angles.cos()) + (_rotate_half(x) * angles.sin())


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = apply_rope(q.transpose(1, 2))
        k = apply_rope(k.transpose(1, 2))
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = self.dropout(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(batch, steps, dim)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, channels: int, samples: int, targets: int = 5, dim: int = 192) -> None:
        super().__init__()
        self.patch = nn.Conv1d(channels, dim, kernel_size=10, stride=5, padding=2)
        self.blocks = nn.Sequential(*(TransformerBlock(dim, heads=4) for _ in range(3)))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, targets))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.patch(x).transpose(1, 2)
        z = self.blocks(z).mean(dim=1)
        return {"pred": self.head(z)}


class NeuroMoE(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        targets: int = 5,
        dim: int = 160,
        experts: int = 6,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels, dim, kernel_size=9, stride=3, padding=4),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, stride=2, padding=2, groups=4),
            nn.GELU(),
        )
        self.shared = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(experts)
        )
        self.router = nn.Linear(dim, experts)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, targets))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.stem(x).transpose(1, 2)
        pooled = tokens.mean(dim=1)
        router_logits = self.router(pooled)
        weights = router_logits.softmax(dim=-1)
        expert_stack = torch.stack([expert(pooled) for expert in self.experts], dim=1)
        expert_mix = (weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        z = self.shared(pooled) + expert_mix
        entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean()
        return {"pred": self.head(z), "router_weights": weights, "router_entropy": entropy}


class HRMRegressor(nn.Module):
    def __init__(
        self,
        channels: int,
        samples: int,
        targets: int = 5,
        dim: int = 192,
        cycles: int = 3,
        low_steps: int = 3,
    ) -> None:
        super().__init__()
        self.cycles = cycles
        self.low_steps = low_steps
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, dim, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.low = nn.GRUCell(dim * 2, dim)
        self.high = nn.GRUCell(dim, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, targets))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.encoder(x).transpose(1, 2)
        context = tokens.mean(dim=1)
        z_high = torch.zeros_like(context)
        z_low = torch.zeros_like(context)
        for _ in range(self.cycles):
            for _ in range(self.low_steps):
                z_low = self.low(torch.cat([context, z_high], dim=-1), z_low)
            z_high = self.high(z_low, z_high)
            z_low = torch.zeros_like(z_low)
        return {"pred": self.head(z_high)}


class HRMMoERegressor(nn.Module):
    def __init__(self, channels: int, samples: int, targets: int = 5, dim: int = 160) -> None:
        super().__init__()
        self.hrm = HRMRegressor(channels, samples, targets=dim, dim=dim, cycles=2, low_steps=2)
        self.moe = NeuroMoE(dim, 1, targets=targets, dim=dim, experts=4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.hrm(x)["pred"].unsqueeze(-1)
        out = self.moe(latent)
        return out


def build_model(name: str, channels: int, samples: int, targets: int = 5) -> nn.Module:
    normalized = name.lower()
    if normalized == "auto":
        normalized = "cnn"
    if normalized == "cnn":
        return LightweightCNN(channels, samples, targets)
    if normalized == "transformer":
        return TinyTransformer(channels, samples, targets)
    if normalized == "neuromoe":
        return NeuroMoE(channels, samples, targets)
    if normalized == "hrm":
        return HRMRegressor(channels, samples, targets)
    if normalized in {"hrm_moe", "hrm-moe", "combined"}:
        return HRMMoERegressor(channels, samples, targets)
    raise ValueError(f"Unknown model '{name}'.")


def load_data(path: str) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    train = TensorDataset(torch.from_numpy(data["x_train"]), torch.from_numpy(data["y_train"]))
    val = TensorDataset(torch.from_numpy(data["x_val"]), torch.from_numpy(data["y_val"]))
    test = TensorDataset(torch.from_numpy(data["x_test"]), torch.from_numpy(data["y_test"]))
    return train, val, test, metadata


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    preds = []
    targets = []
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)["pred"]
        losses.append(torch.nn.functional.mse_loss(pred, y).item())
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())
    pred_np = np.concatenate(preds, axis=0)
    target_np = np.concatenate(targets, axis=0)
    pearson = pearson_per_target(pred_np, target_np)
    rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))
    return {
        "loss": float(np.mean(losses)),
        "rmse": rmse,
        "pearson": pearson,
        "mean_pearson": float(np.mean(pearson)),
        "pred": pred_np,
        "target": target_np,
    }


def train_model(config: TrainConfig) -> dict[str, Any]:
    set_seed(config.seed)
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, metadata = load_data(config.data_path)
    sample_x, sample_y = train_ds[0]
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)

    model = build_model(config.model, sample_x.shape[0], sample_x.shape[1], sample_y.shape[0]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    criterion = nn.MSELoss()

    batch_size = config.batch_size
    if batch_size <= 0:
        batch_size = 64 if device.type == "cuda" else 32

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    start = time.perf_counter()
    deadline = start + config.budget_minutes * 60.0
    history: list[dict[str, float]] = []
    steps = 0
    best_val = -float("inf")
    best_state = None
    router_usage_accum = []
    router_entropy_accum = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_losses = []
        for x, y in train_loader:
            if time.perf_counter() >= deadline and steps > 0:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=config.use_amp and device.type == "cuda"):
                out = model(x)
                loss = criterion(out["pred"], y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            steps += 1

            weights = out.get("router_weights")
            if weights is not None:
                router_usage_accum.append(weights.detach().mean(dim=0).cpu().numpy())
            entropy = out.get("router_entropy")
            if entropy is not None:
                router_entropy_accum.append(float(entropy.detach().cpu()))

        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": float(val_metrics["loss"]),
                "val_mean_pearson": float(val_metrics["mean_pearson"]),
                "val_rmse": float(val_metrics["rmse"]),
            }
        )
        if val_metrics["mean_pearson"] > best_val:
            best_val = float(val_metrics["mean_pearson"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if time.perf_counter() >= deadline and steps > 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - start
    peak_vram = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )

    predictions_path = run_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        val_pred=val_metrics["pred"],
        val_target=val_metrics["target"],
        test_pred=test_metrics["pred"],
        test_target=test_metrics["target"],
    )

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    router_usage = (
        np.mean(np.stack(router_usage_accum), axis=0).tolist() if router_usage_accum else []
    )
    router_entropy = router_entropy_accum
    router_path = run_dir / "router.json"
    router_path.write_text(
        json.dumps({"usage": router_usage, "entropy": router_entropy}, indent=2),
        encoding="utf-8",
    )

    return {
        "model": config.model,
        "seed": config.seed,
        "subject": config.subject,
        "budget_minutes": config.budget_minutes,
        "epochs": len(history),
        "steps": steps,
        "metric": "mean_pearson",
        "mean_pearson": float(val_metrics["mean_pearson"]),
        "per_target_pearson": val_metrics["pearson"].tolist(),
        "rmse": float(val_metrics["rmse"]),
        "test_mean_pearson": float(test_metrics["mean_pearson"]),
        "test_rmse": float(test_metrics["rmse"]),
        "params": count_parameters(model),
        "peak_vram_mb": float(peak_vram),
        "train_seconds": float(elapsed),
        "history": history,
        "metadata": metadata,
        "predictions_path": str(predictions_path),
        "history_path": str(history_path),
        "router_path": str(router_path),
    }
