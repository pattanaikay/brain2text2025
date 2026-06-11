"""
core/episodic_memory.py
-----------------------
ZenBrain episodic memory stage (live) + the episodic-consistency objective, merged
into one standalone module.

  - EpisodicBuffer: fixed-shape ring of past high-confidence latents + a cross-attention
    read head + a learnable fusion gate. Session-keyed + confidence-gated writes.
    forward() emits last_read = {memory_query, memory_retrieved} for the consistency loss.
  - episodic_consistency_loss(query, retrieved): MSE(query, retrieved.detach()) — pulls the
    current latent toward the recalled (stable) past representation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicWritePolicy:
    """Session-keyed + confidence-gated ring write (no grad — buffer is state)."""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    @torch.no_grad()
    def write(self, module: "EpisodicBuffer", latent: torch.Tensor,
              confidence=None, session_id: int | None = None) -> int:
        B = latent.size(0)
        pooled = latent.mean(dim=1).detach()                      # (B, E)
        if confidence is None:
            conf = torch.ones(B, device=latent.device)
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
    def __init__(self, embed_dim: int = 384, buffer_size: int = 256, n_heads: int = 6,
                 confidence_threshold: float = 0.5, gate_init: float = 0.1,
                 retrieval: str = "paramfree", fuse: bool = True, min_read: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.buffer_size = buffer_size
        self.retrieval = retrieval     # "paramfree" (similarity attn, no training) | "learned"
        self.fuse = fuse               # wake read: fuse recall into output (C3b) or keep clean (C3a)
        self.min_read = min_read       # cold-buffer guard: need >= this many entries before reading
        self.register_buffer("buffer", torch.zeros(buffer_size, embed_dim))
        self.register_buffer("buf_session", torch.full((buffer_size,), -1, dtype=torch.long))
        self.register_buffer("write_ptr", torch.zeros(1, dtype=torch.long))
        self.read_head = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        inv = torch.logit(torch.tensor(float(gate_init)).clamp(1e-4, 1 - 1e-4))
        self.gate = nn.Parameter(inv)
        self.write_policy = EpisodicWritePolicy(confidence_threshold)
        self.last_read: dict[str, torch.Tensor] = {}

    def reset(self):
        self.buffer.zero_(); self.buf_session.fill_(-1); self.write_ptr.zero_()

    def is_empty(self) -> bool:
        return bool((self.buf_session < 0).all().item())

    def forward(self, x, write=True, confidence=None, session_id=None):
        """x: (B,T,E) -> (B,T,E). Recall nearest confident past latents from the buffer; fuse
        them into the output only when `self.fuse` (the wake read, C3b). Always exposes
        `last_read` for the sleep-time anchor loss (MSE(query, retrieved.detach()))."""
        occ = self.buf_session >= 0
        if int(occ.sum().item()) < self.min_read:        # cold buffer: nothing reliable to recall
            self.last_read = {"memory_query": x, "memory_retrieved": x}   # -> anchor contributes 0
            if write:
                self._maybe_write(x, confidence, session_id)
            return x
        kv = self.buffer[occ].unsqueeze(0).expand(x.size(0), -1, -1).to(x.dtype)   # (B,K,E)
        if self.retrieval == "learned":
            attended, _ = self.read_head(x, kv, kv)
        else:                                            # parameter-free similarity attention
            scores = torch.matmul(x, kv.transpose(1, 2)) / (self.embed_dim ** 0.5)  # (B,T,K)
            attended = torch.matmul(torch.softmax(scores, dim=-1), kv)              # (B,T,E)
        self.last_read = {"memory_query": x, "memory_retrieved": attended}
        out = x + torch.sigmoid(self.gate) * attended if self.fuse else x
        if write:
            self._maybe_write(x, confidence, session_id)
        return out

    def _maybe_write(self, x, confidence, session_id):
        sid = None
        if isinstance(session_id, (list, tuple)) and session_id:
            try: sid = int(session_id[0])
            except (ValueError, TypeError): sid = None
        elif isinstance(session_id, int):
            sid = session_id
        self.write_policy.write(self, x, confidence=confidence, session_id=sid)


def episodic_consistency_loss(query: torch.Tensor, retrieved: torch.Tensor,
                              weight: float = 1.0) -> torch.Tensor:
    """Recall-the-past-latent: pull current latent toward the (detached) recalled memory."""
    return F.mse_loss(query, retrieved.detach()) * weight
