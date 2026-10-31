"""Classical building blocks: MLP/CNN feature extractors and the
classification head. These are plain PyTorch modules, reused both by the
pure classical baselines and as the "classical half" of a hybrid model.
"""

import torch
from torch import nn

from app.schemas.architecture import CNNExtractorConfig, ClassificationHeadConfig, MLPExtractorConfig

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


def build_mlp_extractor(config: MLPExtractorConfig, input_dim: int) -> tuple[nn.Module, int]:
    layers: list[nn.Module] = []
    dim = input_dim
    activation = _ACTIVATIONS[config.activation]
    for hidden in config.hidden_dims:
        layers.append(nn.Linear(dim, hidden))
        layers.append(activation())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))
        dim = hidden
    return nn.Sequential(*layers), dim


class _FlattenCNN(nn.Module):
    def __init__(self, conv: nn.Module, fc: nn.Module):
        super().__init__()
        self.conv = conv
        self.fc = fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


def build_cnn_extractor(
    config: CNNExtractorConfig, image_height: int, image_width: int, in_channels: int = 1
) -> tuple[nn.Module, int]:
    activation = _ACTIVATIONS[config.activation]
    conv_layers: list[nn.Module] = []
    channels = in_channels
    h, w = image_height, image_width
    padding = config.kernel_size // 2
    for out_channels in config.channels:
        conv_layers.append(nn.Conv2d(channels, out_channels, kernel_size=config.kernel_size, padding=padding))
        conv_layers.append(activation())
        if config.pooling and h >= 2 and w >= 2:
            conv_layers.append(nn.MaxPool2d(2))
            h, w = h // 2, w // 2
        channels = out_channels
    if config.dropout > 0:
        conv_layers.append(nn.Dropout2d(config.dropout))

    conv = nn.Sequential(*conv_layers)
    flat_dim = channels * h * w
    fc = nn.Sequential(nn.Linear(flat_dim, config.fc_dim), activation())
    return _FlattenCNN(conv, fc), config.fc_dim


def build_classification_head(input_dim: int, config: ClassificationHeadConfig) -> nn.Module:
    return nn.Linear(input_dim, config.n_outputs)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
