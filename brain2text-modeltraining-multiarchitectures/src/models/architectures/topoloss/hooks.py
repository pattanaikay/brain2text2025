import torch.nn as nn


def collect_ffn_first_linears(encoder: nn.Module) -> list:
    """Return the first Linear in each FFN/MLP Sequential block of the encoder.

    Targets BIT TransformerBlock.mlp = Sequential(Linear, GELU, Dropout, Linear, Dropout)
    and ConformerBlock FFN modules with the same structure.
    """
    targets = []
    for module in encoder.modules():
        if (
            isinstance(module, nn.Sequential)
            and len(module) >= 1
            and isinstance(module[0], nn.Linear)
        ):
            targets.append(module[0])
    return targets
