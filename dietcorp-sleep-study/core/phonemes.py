"""
core/phonemes.py
----------------
Phoneme inventory (ID <-> ARPAbet) for word-level WER decoding.

IMPORTANT: the *authoritative* mapping is whatever formatted the dataset's `seq_class_ids`.
The trained head has 42 classes = blank(0) + 41 phonemes. After the dataloader's `+1` shift,
stored seq_class_ids are 0..40 (41 classes). The default below is the standard Brain-to-Text /
Willett ARPAbet set (39 CMU phonemes + SIL + an inter-word SP). VERIFY it with the oracle
self-check in run_study (`--validate_mapping`): decoding the ground-truth seq_class_ids to
words must reproduce the transcription (WER ~ 0). If it does not, replace PHONEMES with the
exact list your formatter used (or use DietCorp's own LM decoder — see README).
"""

# 39 CMU phonemes, then SIL, then inter-word SP  -> 41 entries (stored ids 0..40)
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
    "SIL", "SP",
]

# Tokens that act as word boundaries during decoding (not part of any word's pronunciation).
BOUNDARY = {"SIL", "SP"}

# Stored seq_class_ids are 0-indexed into PHONEMES; the model emits ids 1..41 (blank=0),
# i.e. head_id = stored_id + 1.
def head_id_to_arpabet(head_id: int) -> str | None:
    """Map a model phoneme id (1..41; 0=blank) to its ARPAbet symbol."""
    sid = head_id - 1
    if 0 <= sid < len(PHONEMES):
        return PHONEMES[sid]
    return None

def arpabet_to_head_id(sym: str) -> int | None:
    try:
        return PHONEMES.index(sym) + 1
    except ValueError:
        return None

def ids_to_arpabet(ids) -> list[str]:
    out = []
    for i in (ids.tolist() if hasattr(ids, "tolist") else ids):
        s = head_id_to_arpabet(int(i))
        if s is not None:
            out.append(s)
    return out
