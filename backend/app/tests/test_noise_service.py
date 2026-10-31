from app.schemas.architecture import NoiseConfig
from app.services import noise_service


def test_disabled_noise_returns_none():
    assert noise_service.build_noise_model(NoiseConfig(enabled=False), n_qubits=2) is None


def test_enabled_noise_builds_a_model():
    model = noise_service.build_noise_model(
        NoiseConfig(enabled=True, depolarizing_prob=0.02, readout_error_prob=0.02), n_qubits=2
    )
    assert model is not None
