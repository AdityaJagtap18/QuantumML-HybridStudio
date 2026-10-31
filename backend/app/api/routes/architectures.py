from fastapi import APIRouter, HTTPException

from app.schemas.architecture import QuantumLayerConfig
from app.services import quantum_layer_service

router = APIRouter(prefix="/architectures", tags=["architectures"])


@router.post("/circuit-preview")
def preview_circuit(config: QuantumLayerConfig) -> dict:
    try:
        quantum_layer_service.validate_config(config)
        feature_map = quantum_layer_service.build_feature_map(config)
        ansatz = quantum_layer_service.build_ansatz(config)
        combined = quantum_layer_service.build_combined_circuit(feature_map, ansatz)
        summary = quantum_layer_service.summarize(feature_map, ansatz, combined)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "n_qubits": summary.n_qubits,
        "depth": summary.depth,
        "gate_counts": summary.gate_counts,
        "n_encoding_parameters": summary.n_encoding_parameters,
        "n_trainable_parameters": summary.n_trainable_parameters,
        "ascii_drawing": summary.ascii_drawing,
    }
