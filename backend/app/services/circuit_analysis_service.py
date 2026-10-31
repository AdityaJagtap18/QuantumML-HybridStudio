"""Aggregates the full model's resource summary, keeping classical and
quantum parameter counts — and theoretical circuit structure vs. actual
execution measurements — explicitly separate rather than one fused
"complexity" number.
"""

from dataclasses import dataclass

from app.services.hybrid_model_service import ModelBundle
from app.services.quantum_layer_service import CircuitSummary


@dataclass
class ResourceSummary:
    classical_parameters: int
    quantum_parameters: int
    total_parameters: int
    uses_quantum_layer: bool
    circuit: CircuitSummary | None


def build_resource_summary(bundle: ModelBundle) -> ResourceSummary:
    return ResourceSummary(
        classical_parameters=bundle.classical_parameters,
        quantum_parameters=bundle.quantum_parameters,
        total_parameters=bundle.classical_parameters + bundle.quantum_parameters,
        uses_quantum_layer=bundle.quantum_parameters > 0,
        circuit=bundle.quantum_summary,
    )
