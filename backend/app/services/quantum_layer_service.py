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
from qiskit.circuit.library import real_amplitudes, zz_feature_map, z_feature_map
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.circuit.library import efficient_su2
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

from app.schemas.architecture import QuantumLayerConfig


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


def build_quantum_layer(config: QuantumLayerConfig) -> TorchConnector:
    feature_map = build_feature_map(config)
    ansatz = build_ansatz(config)
    combined = build_combined_circuit(feature_map, ansatz)

    if config.qnn_type == "estimator":
        qnn = EstimatorQNN(
            circuit=combined,
            estimator=StatevectorEstimator(),
            input_params=list(feature_map.parameters),
            weight_params=list(ansatz.parameters),
            input_gradients=True,
        )
    else:
        qnn = SamplerQNN(
            circuit=combined,
            sampler=StatevectorSampler(),
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
