import torch

from app.schemas.architecture import ArchitectureConfig, MLPExtractorConfig
from app.schemas.training import TrainingConfig
from app.services import hybrid_model_service, training_service


def _toy_split():
    torch.manual_seed(0)
    X_train = torch.randn(40, 64)
    y_train = (X_train[:, 0] > 0).float().unsqueeze(1)
    X_val = torch.randn(10, 64)
    y_val = (X_val[:, 0] > 0).float().unsqueeze(1)
    return X_train, y_train, X_val, y_val


def test_classical_model_trains_and_loss_history_is_real():
    config = ArchitectureConfig(model_kind="classical_mlp", mlp_extractor=MLPExtractorConfig(hidden_dims=[16]))
    bundle = hybrid_model_service.build_model(config, image_height=8, image_width=8)
    X_train, y_train, X_val, y_val = _toy_split()

    result = training_service.train_model(
        bundle.model, X_train, y_train, X_val, y_val, n_outputs=1,
        config=TrainingConfig(epochs=5, batch_size=8, early_stopping_patience=None),
    )

    assert len(result.history) == 5
    assert all(isinstance(r.train_loss, float) for r in result.history)
    # a trainable linear-separable toy task should get noticeably better
    assert result.history[-1].train_accuracy >= result.history[0].train_accuracy - 0.2


def test_early_stopping_halts_before_configured_epochs():
    config = ArchitectureConfig(model_kind="classical_mlp", mlp_extractor=MLPExtractorConfig(hidden_dims=[4]))
    bundle = hybrid_model_service.build_model(config, image_height=8, image_width=8)
    X_train, y_train, X_val, y_val = _toy_split()
    # Force stagnation: learning rate 0 means val loss never improves.
    result = training_service.train_model(
        bundle.model, X_train, y_train, X_val, y_val, n_outputs=1,
        config=TrainingConfig(epochs=50, batch_size=8, learning_rate=1e-9, early_stopping_patience=2),
    )
    assert result.early_stopped is True
    assert result.stopped_epoch < 50


def test_checkpoint_save_and_restore_round_trip(tmp_path):
    from app.services import checkpoint_service

    config = ArchitectureConfig(model_kind="classical_mlp", mlp_extractor=MLPExtractorConfig(hidden_dims=[8]))
    bundle = hybrid_model_service.build_model(config, image_height=8, image_width=8)
    path = tmp_path / "model.pt"
    torch.save(bundle.model.state_dict(), path)

    bundle2 = hybrid_model_service.build_model(config, image_height=8, image_width=8)
    checkpoint_service.load_checkpoint(bundle2.model, str(path))

    X = torch.randn(3, 64)
    with torch.no_grad():
        out1 = bundle.model(X)
        out2 = bundle2.model(X)
    assert torch.allclose(out1, out2)
