# Arch-5: Audio-LLM swap variants.
# These are full E2E model wrappers, not encoders — not in ENCODER_REGISTRY.
# Import lazily to avoid requiring model weights at import time.

__all__ = ["BrainToTextQwen2Audio", "BrainToTextWhisperQwen", "BrainToTextPhi4MM"]


def __getattr__(name):
    if name == "BrainToTextQwen2Audio":
        from .qwen2_audio_e2e import BrainToTextQwen2Audio
        return BrainToTextQwen2Audio
    if name == "BrainToTextWhisperQwen":
        from .whisper_qwen_e2e import BrainToTextWhisperQwen
        return BrainToTextWhisperQwen
    if name == "BrainToTextPhi4MM":
        from .phi4_mm_e2e import BrainToTextPhi4MM
        return BrainToTextPhi4MM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
