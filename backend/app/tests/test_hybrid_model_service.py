"""The most important test file in this project.

A successful forward pass through the quantum layer is NOT accepted as
proof that the hybrid network is genuinely trainable end to end. These
tests directly verify that gradients reach the quantum layer's
parameters, that they reach the classical layer BEFORE the quantum layer
(proving backprop actually crosses the quantum boundary, not just stops
there), and that an optimizer step measurably changes the quantum
parameters.
"""

import torch
from torch import nn

from app.schemas.architecture import (
    ArchitectureConfig,
    MLPExtractorConfig,
    NoiseConfig,
    ProjectionConfig,
    QuantumLayerConfig,
)
from app.services import hybrid_model_service


def _hybrid_config(qnn_type: str = "estimator", noisy: bool = False) -> ArchitectureConfig:
    return ArchitectureConfig(
        model_kind="hybrid_mlp_qnn",
        mlp_extractor=MLPExtractorConfig(hidden_dims=[8]),
        projection=ProjectionConfig(),
        quantum_layer=QuantumLayerConfig(
            n_qubits=2,
            qnn_type=qnn_type,
            ansatz_reps=1,
            execution_mode="noisy" if noisy else "ideal",
            shots=256,
            noise=NoiseConfig(enabled=True, depolarizing_prob=0.02, readout_error_prob=0.02) if noisy else NoiseConfig(),
        ),
    )


def test_hybrid_model_builds_with_classical_and_quantum_parameters():
    bundle = hybrid_model_service.build_model(_hybrid_config(), image_height=8, image_width=8)
    assert bundle.classical_parameters > 0
    assert bundle.quantum_parameters > 0
    assert bundle.quantum_summary is not None
    assert bundle.quantum_summary.n_trainable_parameters == bundle.quantum_parameters


def test_gradients_reach_the_quantum_layer_and_cross_it_backward():
    bundle = hybrid_model_service.build_model(_hybrid_config(), image_height=8, image_width=8)
    model = bundle.model

    X = torch.randn(6, 64)
    y = torch.randint(0, 2, (6, 1)).float()

    out = model(X)
    loss = nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()

    # model[0] = MLP extractor, model[1] = projection, model[2] = quantum layer (TorchConnector), model[3] = head
    extractor_first_layer = model[0][0]
    quantum_layer = model[2]

    assert extractor_first_layer.weight.grad is not None
    assert float(extractor_first_layer.weight.grad.abs().sum()) > 0.0, (
        "gradient did not propagate backward through the quantum layer to the preceding "
        "classical extractor — forward-pass success alone does not prove end-to-end training"
    )
    assert quantum_layer.weight.grad is not None
    assert float(quantum_layer.weight.grad.abs().sum()) > 0.0


def test_optimizer_step_changes_quantum_parameters():
    bundle = hybrid_model_service.build_model(_hybrid_config(), image_height=8, image_width=8)
    model = bundle.model
    quantum_layer = model[2]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    before = quantum_layer.weight.detach().clone()

    X = torch.randn(6, 64)
    y = torch.randint(0, 2, (6, 1)).float()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(X)
        loss = nn.functional.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

    after = quantum_layer.weight.detach().clone()
    assert not torch.allclose(before, after), "quantum layer parameters did not change after optimizer steps"


def test_sampler_qnn_variant_also_trains_end_to_end():
    bundle = hybrid_model_service.build_model(_hybrid_config(qnn_type="sampler"), image_height=8, image_width=8)
    model = bundle.model
    X = torch.randn(4, 64)
    y = torch.randint(0, 2, (4, 1)).float()

    out = model(X)
    loss = nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()

    assert model[2].weight.grad is not None
    assert float(model[2].weight.grad.abs().sum()) > 0.0


def test_noisy_estimator_qnn_still_trains_end_to_end():
    """The noisy (Aer + synthetic noise model) execution path must be
    just as genuinely trainable as the ideal path -- noise changes the
    simulated readout, not whether gradients propagate."""
    bundle = hybrid_model_service.build_model(_hybrid_config(noisy=True), image_height=8, image_width=8)
    model = bundle.model
    extractor_first_layer = model[0][0]
    quantum_layer = model[2]

    X = torch.randn(4, 64)
    y = torch.randint(0, 2, (4, 1)).float()
    out = model(X)
    loss = nn.functional.binary_cross_entropy_with_logits(out, y)
    loss.backward()

    assert float(extractor_first_layer.weight.grad.abs().sum()) > 0.0
    assert float(quantum_layer.weight.grad.abs().sum()) > 0.0


def test_classical_only_model_has_no_quantum_parameters():
    config = ArchitectureConfig(model_kind="classical_mlp", mlp_extractor=MLPExtractorConfig(hidden_dims=[8]))
    bundle = hybrid_model_service.build_model(config, image_height=8, image_width=8)
    assert bundle.quantum_parameters == 0
    assert bundle.quantum_summary is None
