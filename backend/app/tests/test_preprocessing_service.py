import numpy as np
import pytest

from app.schemas.architecture import PreprocessingConfig
from app.services import preprocessing_service


def _toy_images():
    rng = np.random.default_rng(0)
    X_train = rng.integers(0, 17, size=(20, 8, 8)).astype(float)
    X_val = rng.integers(0, 17, size=(5, 8, 8)).astype(float)
    X_test = rng.integers(0, 17, size=(5, 8, 8)).astype(float)
    return X_train, X_val, X_test


def test_normalize_scales_into_0_1_from_train_pixel_max():
    X_train, X_val, X_test = _toy_images()
    result = preprocessing_service.apply_preprocessing(
        X_train, X_val, X_test, PreprocessingConfig(normalize=True), flatten=False, pixel_max=16.0
    )
    assert result.X_train.max() <= 1.0 + 1e-6
    assert result.X_train.min() >= 0.0


def test_flatten_produces_1d_features():
    X_train, X_val, X_test = _toy_images()
    result = preprocessing_service.apply_preprocessing(
        X_train, X_val, X_test, PreprocessingConfig(normalize=True), flatten=True, pixel_max=16.0
    )
    assert result.X_train.shape == (20, 64)
    assert result.final_dim == 64


def test_pca_reduces_dimension():
    X_train, X_val, X_test = _toy_images()
    result = preprocessing_service.apply_preprocessing(
        X_train, X_val, X_test, PreprocessingConfig(use_pca=True, n_components=4), flatten=False, pixel_max=16.0
    )
    assert result.final_dim == 4
    assert result.original_dim == 64


def test_pca_rejects_too_many_components():
    X_train, X_val, X_test = _toy_images()
    with pytest.raises(ValueError):
        preprocessing_service.apply_preprocessing(
            X_train, X_val, X_test, PreprocessingConfig(use_pca=True, n_components=100), flatten=False, pixel_max=16.0
        )


def test_cnn_path_keeps_2d_shape_without_pca():
    X_train, X_val, X_test = _toy_images()
    result = preprocessing_service.apply_preprocessing(
        X_train, X_val, X_test, PreprocessingConfig(normalize=True, use_pca=False), flatten=False, pixel_max=16.0
    )
    assert result.X_train.shape == (20, 8, 8)
