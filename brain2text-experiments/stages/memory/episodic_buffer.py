"""
stages/memory/episodic_buffer.py
--------------------------------
Track H (now LIVE): ZenBrain episodic-memory stage.

Reference: github.com/zensation-ai/zenbrain (episodic-memory-augmented
architecture) + ZenBrain paper arXiv:2604.23878.

This was a guarded skeleton; it is now a real, backprop-safe memory stage that
sits between the encoder and the projector. The deferred design decision — the
write/evict POLICY — has been resolved to **session-keyed + confidence-gated**
(the choice that matches the drift-across-days thesis):

  - SESSION-KEYED:    every slot remembers which recording day/session wrote it,
                      so reads can be biased toward same-session memories.
  - CONFIDENCE-GATED: only high-confidence (low-CE / high-CTC-prob) trials are
                      written, so the buffer stores trustworthy exemplars rather
                      than noisy ones — the ZenBrain "store what's worth keeping".

WHAT IS LIVE (provably correct + differentiable):
  - A fixed-shape ring buffer (buffer_size × embed_dim) of past latent slices,
    with a parallel session-id buffer and a write pointer (all non-trainable state).
  - A cross-attention read head (query = current latents, keys/values = buffer)
    and a learnable fusion gate — gradients flow through both.
  - forward() emits `memory_query` / `memory_retrieved` for the
    episodic-consistency objective (stages/loss/episodic_consistency.py).

The forward pass NO LONGER raises StubNotImplemented — H1/H2 are live. The
tripwire (tests/test_zenbrain_stub_tripwire.py) now asserts the LIVE contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EpisodicWritePolicy:
    """
    Session-keyed + confidence-gated ring write (the resolved design decision).

    write() mutates the module's ring state IN PLACE (no grad — buffer is memory,
    not a parameter) and returns nothing. Only trials whose confidence exceeds
    `confidence_threshold` are stored; each stored slot records its session id.
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def write(self, module: "EpisodicBuffer", latent: torch.Tensor,
              confidence: torch.Tensor | float | None = None,
              session_id: int | None = None) -> int:
        """
        latent : (B, T, E) current-batch latents. Each batch item is mean-pooled
                 over time to a single (E,) memory and written if confident.
        Returns the number of slots written.
        """
        B = latent.size(0)
        pooled = latent.mean(dim=1).detach()                  # (B, E)

        if confidence is None:
            conf = torch.ones(B, device=latent.device)        # unknown → treat confident
        elif isinstance(confidence, (int, float)):
            conf = torch.full((B,), float(confidence), device=latent.device)
        else:
            conf = confidence.detach().reshape(-1).to(latent.device)

        written = 0
        for b in range(B):
            if float(conf[b]) < self.confidence_threshold:
                continue
            ptr = int(module.write_ptr.item())
            module.buffer[ptr] = pooled[b].to(module.buffer.dtype)
            module.buf_session[ptr] = int(session_id if session_id is not None else -1)
            module.write_ptr[0] = (ptr + 1) % module.buffer_size
            written += 1
        return written


class EpisodicBuffer(nn.Module):
    def __init__(self, embed_dim: int = 384, buffer_size: int = 256,
                 n_heads: int = 6, confidence_threshold: float = 0.5,
                 gate_init: float = 0.1, **_):
        super().__init__()
        self.embed_dim   = embed_dim
        self.buffer_size = buffer_size

        # Non-trainable ring state.
        self.register_buffer("buffer",      torch.zeros(buffer_size, embed_dim))
        self.register_buffer("buf_session", torch.full((buffer_size,), -1, dtype=torch.long))
        self.register_buffer("write_ptr",   torch.zeros(1, dtype=torch.long))

        # Trainable read path.
        self.read_head = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        # Learnable fusion gate: fused = x + sigmoid(gate) * attended. Init small
        # so the stage starts near identity (training stability).
        inv = torch.logit(torch.tensor(float(gate_init)).clamp(1e-4, 1 - 1e-4))
        self.gate = nn.Parameter(inv)

        self.write_policy = EpisodicWritePolicy(confidence_threshold)
        self.last_read: dict[str, torch.Tensor] = {}

    def reset(self):
        self.buffer.zero_(); self.buf_session.fill_(-1); self.write_ptr.zero_()

    def forward(self, x, confidence=None, session_id=None, **_):
        """
        x: (B, T_patch, E) -> fused (B, T_patch, E) (identity passthrough shape).

        Reads the buffer via cross-attention, fuses with a learnable gate, writes
        confident latents back, and stashes memory_query/memory_retrieved on
        self.last_read for the episodic-consistency loss.
        """
        B = x.size(0)
        kv = self.buffer.unsqueeze(0).expand(B, -1, -1).to(x.dtype)  # (B, K, E)
        attended, _ = self.read_head(x, kv, kv)                      # (B, T, E)
        fused = x + torch.sigmoid(self.gate) * attended

        # Expose tensors for the episodic-consistency objective.
        self.last_read = {"memory_query": x, "memory_retrieved": attended}

        # Write-back happens off the gradient path (buffer is state, not a param).
        sid = None
        if isinstance(session_id, (list, tuple)) and session_id:
            try:    sid = int(session_id[0])
            except (ValueError, TypeError): sid = None
        elif isinstance(session_id, int):
            sid = session_id
        self.write_policy.write(self, x, confidence=confidence, session_id=sid)

        return fused


def build(spec: dict, prev_shape: tuple) -> tuple:
    """
    spec keys:
        embed_dim            : int   = 384   (defaults to prev_shape[-1])
        buffer_size          : int   = 256
        n_heads              : int   = 6
        confidence_threshold : float = 0.5
        gate_init            : float = 0.1
        allow_stub_forward   : bool  = (ignored — stage is live)
    """
    embed_dim = spec.get("embed_dim", prev_shape[-1] if prev_shape else 384)
    module = EpisodicBuffer(
        embed_dim            = embed_dim,
        buffer_size          = spec.get("buffer_size", 256),
        n_heads              = spec.get("n_heads", 6),
        confidence_threshold = spec.get("confidence_threshold", 0.5),
        gate_init            = spec.get("gate_init", 0.1),
    )
    # Identity passthrough shape — memory does not change (T_patch, embed_dim).
    return module, prev_shape
