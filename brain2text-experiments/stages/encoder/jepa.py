"""
stages/encoder/jepa.py
----------------------
Track F: JEPA (Joint-Embedding Predictive Architecture) encoder.

Audio-vs-Visual self-supervised pretraining study. The SCIENTIFICALLY
load-bearing part of JEPA — the EMA target-encoder, the stop-gradient, and
masked-latent prediction — is implemented FOR REAL here. The part that
distinguishes "audio" from "visual" is the per-modality PATCHIFIER, now a real
architecture in each case: a single `modality:` spec flag swaps a wav2vec2-style
1D temporal conv stack (audio), a DINOv2-style 2D patch conv (video), or a
native single-layer patch-embed (neural — the control arm). All are random-init
and learned via JEPA; the modality IS the inductive bias under controlled A/B
test. (Pretrained wav2vec2/DINOv2 *weights* are not used: the input is neural
data (B,T,512), not audio/images, so only the architectural bias transfers.)

build(spec, prev_shape) -> (JEPAEncoder, (T_patch, embed_dim))

The encoder satisfies the standard Stack contract: forward(x, ...) returns
context embeddings (B, T_patch, embed_dim) so it drops into encoder→projector
→decoder unchanged. It ADDITIONALLY exposes `pretrain_step(x)` which runs the
real JEPA self-supervised objective and updates the EMA target — that is what
the toy smoke / pretraining loop calls.

Design provenance: ADHD convergence 2026-05-30 (frame: 1-hour-budget +
remove-load-bearing-assumption). "Integration-seam-real / science-stubbed."
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Modality patchifiers (REAL backbones — modality = inductive bias) ─────────
#
# All three consume neural input (B, T, 512) and emit (B, T_patch, embed_dim)
# with T_patch = floor(T / patch_size), so they are drop-in swappable and the
# controlled A/B stays clean (specs differ ONLY in `modality:`). The modality IS
# the architecture. Receptive fields are kept LOCAL on purpose: JEPA masks the
# patchifier outputs (exactly as wav2vec2 masks its conv-encoder outputs), so no
# global mixing may precede the mask or the masked-prediction objective leaks.


class AudioPatchifier1D(nn.Module):
    """wav2vec2-style 1D temporal conv feature-encoder.

    A stack of stride-1 temporal convolutions (local feature extraction over
    time; the 512 neural channels are the conv's input features) followed by a
    strided conv that downsamples to T_patch. This is the wav2vec2 conv-encoder
    inductive bias — deep temporal locality — adapted to neural input."""

    def __init__(self, in_dim: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_dim, embed_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.GELU(),
        )
        self.downsample = nn.Conv1d(embed_dim, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):                        # x: (B, T, C)
        x = x.transpose(1, 2)                    # (B, C, T)
        x = self.features(x)                     # (B, E, T)   local temporal feats
        x = self.downsample(x)                   # (B, E, T_patch)
        return self.norm(x.transpose(1, 2))      # (B, T_patch, E)


class VideoPatchifier2D(nn.Module):
    """DINOv2-style 2D patch embedding.

    Treats (B, T, C) as a single-channel image (B, 1, T, C), applies a local 2D
    convolution (joint time×channel locality — the "seeing" inductive bias),
    then a patch-embed conv that spans the full channel axis and strides
    patch_size on time. 2D spatial locality is what distinguishes the visual
    lens from the audio lens. Local receptive field → JEPA-mask-safe."""

    def __init__(self, in_dim: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=3, padding=1), nn.GELU(),
        )
        # patch-embed: span the full channel axis, stride patch_size on time
        self.patch = nn.Conv2d(
            embed_dim // 4, embed_dim,
            kernel_size=(patch_size, in_dim),
            stride=(patch_size, in_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):                        # x: (B, T, C)
        x = x.unsqueeze(1)                       # (B, 1, T, C)
        x = self.local(x)                        # (B, E/4, T, C)  2D local feats
        x = self.patch(x)                        # (B, E, T_patch, 1)
        x = x.squeeze(-1).transpose(1, 2)        # (B, T_patch, E)
        return self.norm(x)


class NativePatchifier(nn.Module):
    """Provocation arm (neural): a single-layer native patch-embed — no borrowed
    perceptual prior. Models the neural signal on its own terms; the control
    against which the audio/visual inductive biases are judged."""

    def __init__(self, in_dim: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):                        # x: (B, T, C)
        x = x.transpose(1, 2)                    # (B, C, T)
        x = self.proj(x)                         # (B, E, T_patch)
        return self.norm(x.transpose(1, 2))      # (B, T_patch, E)


_PATCHIFIERS = {
    "audio":  AudioPatchifier1D,    # wav2vec2-style temporal conv stack
    "video":  VideoPatchifier2D,    # DINOv2-style 2D patch conv
    "neural": NativePatchifier,     # native single-layer patch-embed (control arm)
}


# ── JEPA encoder ─────────────────────────────────────────────────────────────

class JEPAEncoder(nn.Module):
    """
    Context-encoder + EMA target-encoder + predictor.

    REAL (load-bearing):
        - target_encoder is an EMA copy of the context_encoder, never receives
          gradients (stop-gradient via .detach() on its outputs).
        - pretrain_step() masks a fraction of context patches, predicts the
          target representation at masked positions, returns a smooth-L1 loss.
        - _update_target() does the momentum update.

    REAL backbones:
        - The patchifier per modality is a real architecture (wav2vec2-style 1D
          temporal conv stack / DINOv2-style 2D patch conv / native patch-embed);
          the modality is the inductive bias under controlled A/B test. Random
          init, learned via JEPA (pretrained audio/image weights do not apply to
          neural input).
    NOTE:
        - The shared context-encoder body is a compact TransformerEncoder,
          sufficient for the toy/controlled comparison.
    """

    def __init__(
        self,
        modality: str = "audio",
        input_dim: int = 512,
        embed_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        patch_size: int = 4,
        mask_ratio: float = 0.5,
        ema_momentum: float = 0.996,
        **_ignored,
    ):
        super().__init__()
        if modality not in _PATCHIFIERS:
            raise ValueError(
                f"[jepa] unknown modality '{modality}'. "
                f"Choose one of {list(_PATCHIFIERS)}."
            )
        self.modality     = modality
        self.patch_size   = patch_size          # contrastive.py reads this attr
        self.embed_dim    = embed_dim
        self.mask_ratio   = mask_ratio
        self.ema_momentum = ema_momentum

        patch_cls = _PATCHIFIERS[modality]
        self.patchify = patch_cls(input_dim, embed_dim, patch_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, batch_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # EMA target-encoder: deep copy, frozen, no gradients.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # Predictor maps context reps → predicted target reps.
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    # -- Standard Stack forward: context embeddings for the downstream LLM ----
    def forward(self, x, session_id=None, neural_lengths=None, **_):
        """Returns (B, T_patch, embed_dim) context embeddings.

        This is what encoder→projector→decoder consumes downstream. After JEPA
        pretraining the same context_encoder weights are reused here; no EMA,
        no masking at inference."""
        patches = self.patchify(x)               # (B, T_patch, E)
        return self.context_encoder(patches)

    # -- REAL JEPA self-supervised objective ---------------------------------
    def pretrain_step(self, x):
        """One masked-latent-prediction step. Returns scalar loss.

        Caller is responsible for loss.backward() / optimizer.step(); this
        method also performs the EMA target update under no_grad."""
        patches = self.patchify(x)               # (B, T_patch, E)
        B, T, E = patches.shape

        # Random per-sample mask of context positions.
        n_mask = max(1, int(round(T * self.mask_ratio)))
        mask = torch.zeros(B, T, dtype=torch.bool, device=patches.device)
        for b in range(B):
            idx = torch.randperm(T, device=patches.device)[:n_mask]
            mask[b, idx] = True

        # Context branch sees masked patches zeroed; predicts target reps.
        ctx_in = patches.masked_fill(mask.unsqueeze(-1), 0.0)
        ctx_rep = self.context_encoder(ctx_in)            # grads flow
        pred = self.predictor(ctx_rep)

        # Target branch: full patches, EMA encoder, STOP-GRADIENT.
        with torch.no_grad():
            tgt_rep = self.target_encoder(patches).detach()

        # Loss only on masked positions.
        diff = F.smooth_l1_loss(pred, tgt_rep, reduction="none").mean(-1)  # (B,T)
        loss = (diff * mask).sum() / mask.sum().clamp(min=1)

        self._update_target()
        return loss

    @torch.no_grad()
    def _update_target(self):
        m = self.ema_momentum
        for tp, cp in zip(self.target_encoder.parameters(),
                          self.context_encoder.parameters()):
            tp.mul_(m).add_(cp, alpha=1 - m)


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        modality     : "audio" | "video" | "neural"   (THE A/B switch)
        input_dim    : int   = 512
        embed_dim    : int   = 384
        num_layers   : int   = 4
        num_heads    : int   = 6
        patch_size   : int   = 4
        mask_ratio   : float = 0.5
        ema_momentum : float = 0.996
    """
    patch_size = spec.get("patch_size", 4)
    embed_dim  = spec.get("embed_dim", 384)

    encoder = JEPAEncoder(
        modality     = spec.get("modality", "audio"),
        input_dim    = spec.get("input_dim", 512),
        embed_dim    = embed_dim,
        num_layers   = spec.get("num_layers", 4),
        num_heads    = spec.get("num_heads", 6),
        patch_size   = patch_size,
        mask_ratio   = spec.get("mask_ratio", 0.5),
        ema_momentum = spec.get("ema_momentum", 0.996),
    )

    # Downstream fine-tune: load a JEPA-pretrained backbone if pointed at one.
    # (The pretraining path in run.py saves "encoder."-prefixed weights.)
    ckpt_path = spec.get("pretrained_ckpt")
    if ckpt_path:
        import os
        if os.path.exists(str(ckpt_path)):
            sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            clean = {(k[len("encoder."):] if k.startswith("encoder.") else k): v
                     for k, v in sd.items()}
            encoder.load_state_dict(clean, strict=False)

    T_bins  = prev_shape[0] if prev_shape else 240
    T_patch = math.ceil(T_bins / patch_size)
    return encoder, (T_patch, embed_dim)
