"""Composes the full model: image -> classical extractor -> [feature
projection -> quantum layer] -> classification head. The quantum layer is
a genuine nn.Module in this Sequential, not a separately-trained
component whose output is appended afterward — one forward pass, one
loss, one backward pass, one optimizer step covers everything.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from app.schemas.architecture import ArchitectureConfig, ProjectionConfig
from app.services import classical_model_service, quantum_layer_service

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}


class FeatureProjection(nn.Module):
    """Linear projection into the quantum layer's input dimension. When
    `normalize` is set, the activation's output (bounded by tanh/sigmoid)
    is scaled toward [-pi, pi], a sensible range for rotation-angle
    feature encoding."""

    def __init__(self, in_dim: int, out_dim: int, config: ProjectionConfig):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = _ACTIVATIONS[config.activation]()
        self.scale = math.pi if config.normalize else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x)) * self.scale


@dataclass
class ModelBundle:
    model: nn.Module
    classical_parameters: int
    quantum_parameters: int
    input_kind: str  # "flat" or "image"
    quantum_summary: quantum_layer_service.CircuitSummary | None


def build_model(
    config: ArchitectureConfig, image_height: int, image_width: int, flat_input_dim: int | None = None
) -> ModelBundle:
    n_outputs = config.head.n_outputs

    if config.uses_cnn:
        extractor, extractor_out_dim = classical_model_service.build_cnn_extractor(
            config.cnn_extractor, image_height, image_width
        )
        input_kind = "image"
    else:
        extractor, extractor_out_dim = classical_model_service.build_mlp_extractor(
            config.mlp_extractor, flat_input_dim or (image_height * image_width)
        )
        input_kind = "flat"

    quantum_summary = None
    if config.uses_quantum_layer:
        quantum_layer_service.validate_config(config.quantum_layer)
        projection = FeatureProjection(extractor_out_dim, config.quantum_layer.n_qubits, config.projection)
        q_layer = quantum_layer_service.build_quantum_layer(config.quantum_layer)
        head_in_dim = quantum_layer_service.quantum_output_dim(config.quantum_layer)
        head = classical_model_service.build_classification_head(head_in_dim, config.head)
        model = nn.Sequential(extractor, projection, q_layer, head)

        fm = quantum_layer_service.build_feature_map(config.quantum_layer)
        ansatz = quantum_layer_service.build_ansatz(config.quantum_layer)
        combined = quantum_layer_service.build_combined_circuit(fm, ansatz)
        quantum_summary = quantum_layer_service.summarize(fm, ansatz, combined)

        classical_params = classical_model_service.count_parameters(extractor)
        classical_params += classical_model_service.count_parameters(projection)
        classical_params += classical_model_service.count_parameters(head)
        quantum_params = sum(p.numel() for p in q_layer.parameters() if p.requires_grad)
    else:
        head = classical_model_service.build_classification_head(extractor_out_dim, config.head)
        model = nn.Sequential(extractor, head)
        classical_params = classical_model_service.count_parameters(model)
        quantum_params = 0

    return ModelBundle(
        model=model,
        classical_parameters=classical_params,
        quantum_parameters=quantum_params,
        input_kind=input_kind,
        quantum_summary=quantum_summary,
    )


def prepare_input(X: torch.Tensor | np.ndarray, input_kind: str) -> torch.Tensor:
    tensor = torch.as_tensor(X, dtype=torch.float32)
    if input_kind == "image":
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)  # (N, H, W) -> (N, 1, H, W)
    else:
        if tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
    return tensor
