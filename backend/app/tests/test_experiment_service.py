from app.core.database import init_db
from app.schemas.architecture import ArchitectureConfig, MLPExtractorConfig, PreprocessingConfig, QuantumLayerConfig
from app.schemas.dataset import DatasetConfigRequest
from app.schemas.experiment import ExperimentCreateRequest
from app.schemas.training import TrainingConfig
from app.services import experiment_service

init_db()


def _small_request(name: str, model_kind: str) -> ExperimentCreateRequest:
    return ExperimentCreateRequest(
        name=name,
        dataset=DatasetConfigRequest(dataset_id="digits", selected_classes=[0, 1], max_samples=40, random_seed=0),
        preprocessing=PreprocessingConfig(normalize=True),
        architecture=ArchitectureConfig(
            model_kind=model_kind,
            mlp_extractor=MLPExtractorConfig(hidden_dims=[8]),
            quantum_layer=QuantumLayerConfig(n_qubits=2, ansatz_reps=1),
        ),
        training=TrainingConfig(epochs=2, batch_size=8, early_stopping_patience=None),
    )


def test_classical_experiment_completes():
    record = experiment_service.create_experiment(_small_request("classical-lifecycle", "classical_mlp"))
    experiment_service.run_experiment(record.id)
    finished = experiment_service.get_experiment(record.id)
    assert finished.status == "COMPLETED"
    assert finished.result["resource_summary"]["uses_quantum_layer"] is False
    assert len(finished.training_history) == 2
    assert len(finished.result["predictions"]) > 0


def test_hybrid_experiment_completes_and_reports_quantum_resources():
    record = experiment_service.create_experiment(_small_request("hybrid-lifecycle", "hybrid_mlp_qnn"))
    experiment_service.run_experiment(record.id)
    finished = experiment_service.get_experiment(record.id)
    assert finished.status == "COMPLETED"
    assert finished.result["resource_summary"]["uses_quantum_layer"] is True
    assert finished.result["resource_summary"]["quantum_parameters"] > 0
    assert finished.result["resource_summary"]["circuit"]["n_qubits"] == 2


def test_invalid_pca_component_count_fails_gracefully():
    request = _small_request("invalid-lifecycle", "hybrid_mlp_qnn")
    request.preprocessing = PreprocessingConfig(use_pca=True, n_components=1000)
    record = experiment_service.create_experiment(request)
    experiment_service.run_experiment(record.id)
    finished = experiment_service.get_experiment(record.id)
    assert finished.status == "FAILED"
    assert finished.error is not None


def test_list_and_delete_experiment():
    record = experiment_service.create_experiment(_small_request("to-delete", "classical_mlp"))
    ids = [r.id for r in experiment_service.list_experiments()]
    assert record.id in ids
    assert experiment_service.delete_experiment(record.id) is True
    assert experiment_service.get_experiment(record.id) is None
