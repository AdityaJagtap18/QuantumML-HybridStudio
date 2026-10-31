"""Image preprocessing: normalization and optional PCA-based feature
reduction, always fit on the training split only. Flattening happens
here for the MLP/PCA path; the CNN path keeps the 2D image shape and
normalizes in place.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from app.schemas.architecture import PreprocessingConfig


@dataclass
class PreprocessedImages:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    original_dim: int
    final_dim: int
    flattened: bool
    pipeline_description: list[str]


def _normalize(images: np.ndarray, pixel_max: float) -> np.ndarray:
    return images.astype(np.float32) / max(pixel_max, 1e-8)


def apply_preprocessing(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    config: PreprocessingConfig,
    flatten: bool,
    pixel_max: float,
) -> PreprocessedImages:
    pipeline: list[str] = []
    original_dim = int(np.prod(X_train.shape[1:]))

    if config.normalize:
        X_train = _normalize(X_train, pixel_max)
        X_val = _normalize(X_val, pixel_max)
        X_test = _normalize(X_test, pixel_max)
        pipeline.append(f"Normalize pixels by /{pixel_max:g} (fit range from train)")
    else:
        X_train, X_val, X_test = X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)

    if flatten or config.use_pca:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        pipeline.append("Flatten to 1D feature vector")

    if config.use_pca:
        n_components = config.n_components or min(X_train.shape[1], 4)
        if n_components > original_dim:
            raise ValueError(
                f"n_components ({n_components}) cannot exceed original feature count ({original_dim})"
            )
        pca = PCA(n_components=n_components, random_state=0)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        pipeline.append(f"PCA -> {n_components} components (fit on train)")

    final_dim = int(np.prod(X_train.shape[1:]))

    return PreprocessedImages(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        original_dim=original_dim,
        final_dim=final_dim,
        flattened=flatten or config.use_pca,
        pipeline_description=pipeline,
    )
