"""Builds the differentiable quantum layer: a feature-map + ansatz circuit
wrapped by an EstimatorQNN/SamplerQNN, wrapped again by TorchConnector so
it behaves as an ordinary trainable torch.nn.Module.

`input_gradients=True` is required and non-default — without it,
gradients cannot propagate backward through this layer to whatever
classical layer feeds it (verified empirically; see
docs/IMPLEMENTATION_PLAN.md).
"""

from dataclasses import dataclass

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import efficient_su2, real_amplitudes, z_feature_map, zz_feature_map
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

from app.schemas.architecture import QuantumLayerConfig
from app.services import noise_service


def build_feature_map(config: QuantumLayerConfig) -> QuantumCircuit:
    if config.feature_map == "ZFeatureMap":
        return z_feature_map(feature_dimension=config.n_qubits, reps=config.feature_map_reps)
    return zz_feature_map(
        feature_dimension=config.n_qubits, reps=config.feature_map_reps, entanglement=config.entanglement
    )


def build_ansatz(config: QuantumLayerConfig) -> QuantumCircuit:
    if config.ansatz == "EfficientSU2":
        return efficient_su2(num_qubits=config.n_qubits, reps=config.ansatz_reps, entanglement=config.entanglement)
    return real_amplitudes(num_qubits=config.n_qubits, reps=config.ansatz_reps, entanglement=config.entanglement)


def build_combined_circuit(feature_map: QuantumCircuit, ansatz: QuantumCircuit) -> QuantumCircuit:
    combined = QuantumCircuit(feature_map.num_qubits)
    combined.compose(feature_map, inplace=True)
    combined.compose(ansatz, inplace=True)
    return combined


@dataclass
class CircuitSummary:
    n_qubits: int
    depth: int
    gate_counts: dict[str, int]
    n_encoding_parameters: int
    n_trainable_parameters: int
    ascii_drawing: str


def summarize(feature_map: QuantumCircuit, ansatz: QuantumCircuit, combined: QuantumCircuit) -> CircuitSummary:
    decomposed = combined.decompose()
    return CircuitSummary(
        n_qubits=combined.num_qubits,
        depth=decomposed.depth(),
        gate_counts=dict(decomposed.count_ops()),
        n_encoding_parameters=feature_map.num_parameters,
        n_trainable_parameters=ansatz.num_parameters,
        ascii_drawing=str(combined.draw(output="text", fold=100)),
    )


def _build_estimator(config: QuantumLayerConfig):
    """Ideal uses the exact StatevectorEstimator (no sampling noise at
    all); noisy uses Aer's EstimatorV2 with a synthetic noise model and a
    precision derived from the configured shot count — confirmed
    empirically that AerEstimatorV2 takes `default_precision` (not
    `default_shots`, unlike AerSampler) via its `options` dict."""
    if config.execution_mode == "ideal":
        return StatevectorEstimator()
    noise_model = noise_service.build_noise_model(config.noise, config.n_qubits)
    options = {"default_precision": 1.0 / max(config.shots, 1) ** 0.5}
    if noise_model is not None:
        options["backend_options"] = {"noise_model": noise_model}
    return AerEstimator(options=options)


def _build_sampler(config: QuantumLayerConfig):
    if config.execution_mode == "ideal":
        return StatevectorSampler()
    noise_model = noise_service.build_noise_model(config.noise, config.n_qubits)
    options = {"backend_options": {"noise_model": noise_model}} if noise_model is not None else {}
    return AerSampler(default_shots=config.shots, options=options)


def build_quantum_layer(config: QuantumLayerConfig) -> TorchConnector:
    feature_map = build_feature_map(config)
    ansatz = build_ansatz(config)
    combined = build_combined_circuit(feature_map, ansatz)

    if config.qnn_type == "estimator":
        qnn = EstimatorQNN(
            circuit=combined,
            estimator=_build_estimator(config),
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
            input_gradients=True,
        )
    else:
        qnn = SamplerQNN(
            circuit=combined,
            sampler=_build_sampler(config),
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
            interpret=lambda x: x % 2,
            output_shape=2,
            input_gradients=True,
        )

    return TorchConnector(qnn)


def quantum_output_dim(config: QuantumLayerConfig) -> int:
    """EstimatorQNN (default single observable) outputs one scalar per
    sample; the SamplerQNN configuration here interprets measurement
    outcomes into 2 classes, outputting a 2-way probability vector —
    confirmed empirically, not assumed."""
    return 1 if config.qnn_type == "estimator" else 2


def validate_config(config: QuantumLayerConfig) -> None:
    if config.ansatz == "EfficientSU2" and config.n_qubits < 1:
        raise ValueError("Quantum layer requires at least 1 qubit")
