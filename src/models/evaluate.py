"""Evaluation metrics, resilience index and classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def report_metrics(y_true, y_pred) -> dict[str, float]:
    """Print and return R^2, RMSE and MAE."""
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    print(f"R2   = {metrics['r2']:.4f}")
    print(f"RMSE = {metrics['rmse']:.2f}")
    print(f"MAE  = {metrics['mae']:.2f}")
    return metrics


def resilience_index(actual, predicted) -> np.ndarray:
    """Resilience index = actual yield / predicted yield.

    Values close to 1.0 mean the crop performed as the model expects;
    values well below 0.7 indicate a shortfall (vulnerable).
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return actual / np.where(predicted == 0, np.nan, predicted)


def resilience_class(x, high: float = 0.9, moderate: float = 0.7) -> str:
    """Map a resilience index to a class label."""
    if x >= high:
        return "Highly Resilient"
    if x >= moderate:
        return "Moderately Resilient"
    return "Vulnerable"