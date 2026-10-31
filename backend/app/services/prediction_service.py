"""Per-image predictions for the Prediction Analysis UI — every value
here comes from an actual forward pass through the trained model, never
simulated or guessed."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class ImagePrediction:
    index: int
    true_label: int
    predicted_label: int
    confidence: float
    correct: bool
    pixels: list[list[float]]


def build_predictions(
    model: nn.Module,
    X_test: torch.Tensor,
    raw_images: np.ndarray,
    y_test: np.ndarray,
    n_outputs: int,
    max_items: int = 200,
) -> list[ImagePrediction]:
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        if n_outputs == 1:
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs > 0.5).long()
            confidences = torch.where(preds == 1, probs, 1 - probs)
        else:
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            confidences = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

    n = min(max_items, X_test.shape[0])
    results = []
    for i in range(n):
        true_label = int(y_test[i])
        predicted_label = int(preds[i])
        results.append(
            ImagePrediction(
                index=i,
                true_label=true_label,
                predicted_label=predicted_label,
                confidence=float(confidences[i]),
                correct=true_label == predicted_label,
                pixels=raw_images[i].tolist(),
            )
        )
    return results
