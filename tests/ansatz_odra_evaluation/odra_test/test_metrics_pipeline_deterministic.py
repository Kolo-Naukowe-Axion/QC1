"""Sklearn metrics + thresholding are deterministic for fixed arrays."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score


def predictions_to_labels(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions).reshape(-1)
    return np.where(predictions > 0, 1, -1).astype(np.float32)


def test_metrics_repeatable():
    rng = np.random.default_rng(1)
    y_true = rng.choice([-1.0, 1.0], size=200).astype(np.float32)
    scores = rng.standard_normal(200).astype(np.float32)
    y_pred = predictions_to_labels(scores)
    a1 = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, pos_label=1))
    for _ in range(50):
        assert float(accuracy_score(y_true, y_pred)) == a1
        assert float(f1_score(y_true, y_pred, pos_label=1)) == f1


def test_fold1_labels_file_if_present(project_root):
    path = project_root / "cross_validation" / "fold_1" / "test_data.csv"
    if not path.is_file():
        pytest.skip("fold_1 test_data.csv not in tree")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    y = data[:, 5].astype(np.float32)
    scores = np.linspace(-1, 1, len(y))
    pred = predictions_to_labels(scores)
    _ = float(accuracy_score(y, pred))
    _ = float(f1_score(y, pred, pos_label=1))
