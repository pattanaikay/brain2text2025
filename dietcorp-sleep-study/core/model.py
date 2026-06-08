"""
core/model.py
-------------
CTCPhonemeModel = BIT_Transformer (per-session read-in) + Linear head (42 classes).
This is the exact model produced by train_ctc.py; best_model_per.pth is its state_dict
(`encoder.*` + `head.weight/bias`). Provides loaders that rebuild the right session
read-in set from the checkpoint so weights load cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.bit_encoder import BIT_Transformer


class CTCPhonemeModel(nn.Module):
    """encoder -> CTC phoneme logits. 0 = blank, 1..41 = phonemes."""

    def __init__(self, encoder: BIT_Transformer, num_phonemes: int = 42):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_phonemes)

    def encode(self, x, session_id=None, neural_lengths=None):
        return self.encoder(x, session_id=session_id, neural_lengths=neural_lengths)

    def forward(self, x, session_id=None, neural_lengths=None):
        return self.head(self.encode(x, session_id=session_id, neural_lengths=neural_lengths))


def session_ids_from_checkpoint(state_dict: dict) -> list[str]:
    """Recover the read-in session keys baked into the checkpoint (excluding 'default')."""
    sids = set()
    for k in state_dict:
        # keys look like: encoder.read_in.t15_2023_08_11.weight
        if k.startswith("encoder.read_in.") and k.endswith(".weight"):
            sid = k[len("encoder.read_in."):-len(".weight")]
            if sid != "default":
                sids.add(sid)
    return sorted(sids)


def build_ctc_model(ckpt_path: str, device: torch.device | str = "cpu",
                    patch_size: int = 4, num_phonemes: int = 42) -> CTCPhonemeModel:
    """
    Rebuild CTCPhonemeModel with the checkpoint's session read-in set and load weights.
    Returns the model on `device`, in eval() mode.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    session_ids = session_ids_from_checkpoint(state)
    encoder = BIT_Transformer(session_ids=session_ids, patch_size=patch_size)
    model = CTCPhonemeModel(encoder, num_phonemes=num_phonemes)

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_loaded = len(state) - len(unexpected)
    print(f"[model] loaded {n_loaded}/{len(state)} tensors "
          f"(sessions={len(session_ids)}, missing={len(missing)}, unexpected={len(unexpected)})")
    return model.to(device).eval()


def patch_embed_params(model: CTCPhonemeModel,
                       name_hints=("patch", "read_in")) -> list[nn.Parameter]:
    """
    DietCorp consolidates only the input patch-embedding (+ per-session read-in).
    Select those params by name from the encoder; everything else stays frozen.
    """
    chosen = [p for n, p in model.encoder.named_parameters()
              if any(h in n.lower() for h in name_hints)]
    if not chosen:                       # safety: fall back to all encoder params
        chosen = list(model.encoder.parameters())
    return chosen
