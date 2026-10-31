"""Synthetic Aer noise models — never hardware-calibrated.

Depolarizing error on single/two-qubit gates plus a readout (measurement)
bit-flip error, both with user-configurable probabilities. This is a
configurable simulation knob, not a model of any real device.
"""

from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from app.schemas.architecture import NoiseConfig


def build_noise_model(config: NoiseConfig, n_qubits: int) -> NoiseModel | None:
    if not config.enabled:
        return None

    noise_model = NoiseModel()

    if config.depolarizing_prob > 0:
        error_1q = depolarizing_error(config.depolarizing_prob, 1)
        error_2q = depolarizing_error(config.depolarizing_prob, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["u", "r", "ry", "rz", "x", "h"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx"])

    if config.readout_error_prob > 0:
        p = config.readout_error_prob
        readout_error = ReadoutError([[1 - p, p], [p, 1 - p]])
        for qubit in range(n_qubits):
            noise_model.add_readout_error(readout_error, [qubit])

    return noise_model
