from src.models.encoder import BIT_Transformer
from src.models.architectures.conformer import ConformerEncoder
from src.models.architectures.mamba_possm import MambaPOSSMEncoder
from src.models.architectures.zenbrain_memory import ZenBrainEncoder
from src.models.architectures.moe import MoEEncoder
from src.models.architectures.hrm import HRMEncoder

ENCODER_REGISTRY = {
    "bit":       BIT_Transformer,
    "conformer": ConformerEncoder,
    "mamba":     MambaPOSSMEncoder,
    "zenbrain":  ZenBrainEncoder,
    "moe":       MoEEncoder,
    "hrm":       HRMEncoder,
}


def build_encoder(name: str, **kwargs):
    if name not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}"
        )
    return ENCODER_REGISTRY[name](**kwargs)
