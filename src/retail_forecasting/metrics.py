"""Forecast evaluation metrics."""
from __future__ import annotations

import numpy as np


def calculate_metrics(actual, predicted) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"Actual and predicted arrays must match; got {y.shape} and {p.shape}.")
    if y.size == 0:
        raise ValueError("Cannot calculate metrics on an empty dataset.")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Actual and predicted values must be finite.")
    error = y - p
    denominator = np.abs(y) + np.abs(p)
    smape_terms = np.divide(
        2 * np.abs(error), denominator, out=np.zeros_like(denominator), where=denominator != 0
    )
    total_actual = np.abs(y).sum()
    wape = float(np.abs(error).sum() / total_actual * 100) if total_actual else 0.0
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "wape_pct": wape,
        "smape_pct": float(smape_terms.mean() * 100),
        "bias": float((p - y).mean()),
    }
