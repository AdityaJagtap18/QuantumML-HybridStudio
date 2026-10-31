"""Orchestrates the full experiment lifecycle and its persistence.

QUEUED -> VALIDATING -> PREPROCESSING -> BUILDING_MODEL -> TRAINING ->
EVALUATING -> COMPLETED (or FAILED).
"""

from datetime import datetime, timezone

import numpy as np
import qiskit
import qiskit_machine_learning
import sklearn
import torch
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.experiment import ExperimentRecord
from app.schemas.experiment import ExperimentCreateRequest
from app.services import (
    checkpoint_service,
    circuit_analysis_service,
    evaluation_service,
    hybrid_model_service,
    image_dataset_service,
    prediction_service,
    preprocessing_service,
    training_service,
)

_SOFTWARE_VERSIONS = {
    "torch": torch.__version__,
    "qiskit": qiskit.__version__,
    "qiskit_machine_learning": qiskit_machine_learning.__version__,
    "scikit_learn": sklearn.__version__,
    "numpy": np.__version__,
}


def create_experiment(request: ExperimentCreateRequest) -> ExperimentRecord:
    db = SessionLocal()
    try:
        record = ExperimentRecord(
            name=request.name,
            status="QUEUED",
            stage="QUEUED",
            dataset_id=request.dataset.dataset_id,
            architecture_kind=request.architecture.model_kind,
            config=request.model_dump(mode="json"),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        db.expunge(record)
        return record
    finally:
        db.close()


def get_experiment(experiment_id: str) -> ExperimentRecord | None:
    db = SessionLocal()
    try:
        record = db.get(ExperimentRecord, experiment_id)
        if record:
            db.expunge(record)
        return record
    finally:
        db.close()


def list_experiments() -> list[ExperimentRecord]:
    db = SessionLocal()
    try:
        records = db.query(ExperimentRecord).order_by(ExperimentRecord.created_at.desc()).all()
        for r in records:
            db.expunge(r)
        return records
    finally:
        db.close()


def delete_experiment(experiment_id: str) -> bool:
    db = SessionLocal()
    try:
        record = db.get(ExperimentRecord, experiment_id)
        if not record:
            return False
        db.delete(record)
        db.commit()
        return True
    finally:
        db.close()


def _update(db: Session, record: ExperimentRecord, **fields) -> None:
    for key, value in fields.items():
        setattr(record, key, value)
    db.add(record)
    db.commit()


def run_experiment(experiment_id: str) -> None:
    db = SessionLocal()
    try:
        record = db.get(ExperimentRecord, experiment_id)
        if record is None:
            return
        request = ExperimentCreateRequest.model_validate(record.config)
        _update(db, record, status="RUNNING", stage="VALIDATING", started_at=datetime.now(timezone.utc))

        bundle = image_dataset_service.load_dataset(request.dataset)
        pixel_max = float(bundle.images.max())
        X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = image_dataset_service.split_dataset(
            bundle, request.dataset
        )

        _update(db, record, stage="PREPROCESSING")
        flatten = not request.architecture.uses_cnn
        preprocessed = preprocessing_service.apply_preprocessing(
            X_train_raw, X_val_raw, X_test_raw, request.preprocessing, flatten=flatten, pixel_max=pixel_max
        )

        _update(db, record, stage="BUILDING_MODEL")
        flat_dim = preprocessed.final_dim if flatten else None
        model_bundle = hybrid_model_service.build_model(
            request.architecture, bundle.info.image_height, bundle.info.image_width, flat_input_dim=flat_dim
        )

        X_train_t = hybrid_model_service.prepare_input(preprocessed.X_train, model_bundle.input_kind)
        X_val_t = hybrid_model_service.prepare_input(preprocessed.X_val, model_bundle.input_kind)
        X_test_t = hybrid_model_service.prepare_input(preprocessed.X_test, model_bundle.input_kind)
        y_train_t = torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_val_t = torch.as_tensor(y_val, dtype=torch.float32).unsqueeze(1)

        _update(db, record, stage="TRAINING")
        local_history: list[dict] = []

        def on_epoch(epoch_record):
            local_history.append(
                {
                    "epoch": epoch_record.epoch,
                    "train_loss": epoch_record.train_loss,
                    "val_loss": epoch_record.val_loss,
                    "train_accuracy": epoch_record.train_accuracy,
                    "val_accuracy": epoch_record.val_accuracy,
                    "learning_rate": epoch_record.learning_rate,
                    "duration_seconds": epoch_record.duration_seconds,
                }
            )
            _update(db, record, training_history=list(local_history))

        def do_checkpoint():
            checkpoint_service.save_checkpoint(model_bundle.model, experiment_id)

        training_result = training_service.train_model(
            model_bundle.model,
            X_train_t,
            y_train_t,
            X_val_t,
            y_val_t,
            n_outputs=request.architecture.head.n_outputs,
            config=request.training,
            checkpoint_fn=do_checkpoint,
            on_epoch=on_epoch,
        )
        checkpoint_path = checkpoint_service.checkpoint_path(experiment_id)

        _update(db, record, stage="EVALUATING")
        model_bundle.model.eval()
        with torch.no_grad():
            test_logits = model_bundle.model(X_test_t)
            if request.architecture.head.n_outputs == 1:
                test_probs = torch.sigmoid(test_logits).squeeze(1)
                test_preds = (test_probs > 0.5).long().numpy()
                test_scores = test_probs.numpy()
            else:
                test_probs = torch.softmax(test_logits, dim=1)
                test_preds = test_probs.argmax(dim=1).numpy()
                test_scores = test_probs[:, 1].numpy()

        class_names = bundle.class_names
        eval_result = evaluation_service.evaluate(y_test, test_preds, test_scores, class_names)

        predictions = prediction_service.build_predictions(
            model_bundle.model, X_test_t, X_test_raw, y_test, request.architecture.head.n_outputs
        )

        resource_summary = circuit_analysis_service.build_resource_summary(model_bundle)

        result = {
            "dataset": {
                "id": bundle.info.id,
                "name": bundle.info.name,
                "image_height": bundle.info.image_height,
                "image_width": bundle.info.image_width,
                "n_samples": int(bundle.images.shape[0]),
                "class_names": class_names,
                "train_size": int(X_train_raw.shape[0]),
                "val_size": int(X_val_raw.shape[0]),
                "test_size": int(X_test_raw.shape[0]),
                "random_seed": request.dataset.random_seed,
            },
            "preprocessing": {
                "pipeline": preprocessed.pipeline_description,
                "original_dim": preprocessed.original_dim,
                "final_dim": preprocessed.final_dim,
                "flattened": preprocessed.flattened,
            },
            "architecture": {
                "model_kind": request.architecture.model_kind,
                "uses_quantum_layer": request.architecture.uses_quantum_layer,
                "uses_cnn": request.architecture.uses_cnn,
            },
            "resource_summary": {
                "classical_parameters": resource_summary.classical_parameters,
                "quantum_parameters": resource_summary.quantum_parameters,
                "total_parameters": resource_summary.total_parameters,
                "uses_quantum_layer": resource_summary.uses_quantum_layer,
                "circuit": None
                if resource_summary.circuit is None
                else {
                    "n_qubits": resource_summary.circuit.n_qubits,
                    "depth": resource_summary.circuit.depth,
                    "gate_counts": resource_summary.circuit.gate_counts,
                    "n_encoding_parameters": resource_summary.circuit.n_encoding_parameters,
                    "n_trainable_parameters": resource_summary.circuit.n_trainable_parameters,
                    "ascii_drawing": resource_summary.circuit.ascii_drawing,
                },
            },
            "training": {
                "optimizer": request.training.optimizer,
                "configured_epochs": request.training.epochs,
                "stopped_epoch": training_result.stopped_epoch,
                "early_stopped": training_result.early_stopped,
                "best_val_loss": training_result.best_val_loss,
                "total_seconds": training_result.total_seconds,
                "random_seed": request.training.random_seed,
            },
            "metrics": eval_result.__dict__,
            "predictions": [p.__dict__ for p in predictions],
            "checkpoint_path": checkpoint_path,
            "software_versions": _SOFTWARE_VERSIONS,
        }

        _update(
            db,
            record,
            status="COMPLETED",
            stage="COMPLETED",
            result=result,
            training_history=local_history,
            checkpoint_path=checkpoint_path,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        db.rollback()
        record = db.get(ExperimentRecord, experiment_id)
        if record is not None:
            _update(
                db,
                record,
                status="FAILED",
                stage="FAILED",
                error=str(exc),
                completed_at=datetime.now(timezone.utc),
            )
    finally:
        db.close()
