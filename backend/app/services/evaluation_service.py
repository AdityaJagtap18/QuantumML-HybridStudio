"""Classification metrics computed from actual model predictions — never
hardcoded. Binary by default (digit subsets); macro-averaging handles the
multiclass case for forward compatibility.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    mcc: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: list[list[int]]
    roc_curve: dict[str, list[float]] | None
    pr_curve: dict[str, list[float]] | None
    per_class: dict[str, dict[str, float]]


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray, class_names: list[str]
) -> EvaluationResult:
    is_binary = len(np.unique(y_true)) == 2
    average = "binary" if is_binary else "macro"

    accuracy = float(np.mean(y_true == y_pred))
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    labels = sorted(np.unique(y_true))
    per_class = {
        class_names[int(label)]: {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
        }
        for i, label in enumerate(labels)
    }

    roc_auc = pr_auc = None
    roc_curve_data = pr_curve_data = None
    if is_binary:
        try:
            roc_auc = float(roc_auc_score(y_true, scores))
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

            pr_auc = float(average_precision_score(y_true, scores))
            prec, rec, _ = precision_recall_curve(y_true, scores)
            pr_curve_data = {"precision": prec.tolist(), "recall": rec.tolist()}
        except ValueError:
            pass

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=balanced_acc,
        mcc=mcc,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
        roc_curve=roc_curve_data,
        pr_curve=pr_curve_data,
        per_class=per_class,
    )
