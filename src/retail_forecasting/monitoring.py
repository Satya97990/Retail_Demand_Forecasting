"""Lightweight data-drift and prediction-health monitoring utilities."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from retail_forecasting.config import DEFAULT_TRAIN_PATH, FEATURE_COLUMNS, REPORTS_DIR
from retail_forecasting.data import load_train
from retail_forecasting.features import build_training_features

LOGGER = logging.getLogger(__name__)


def population_stability_index(reference, current, bins: int = 10) -> float:
    """Calculate PSI using quantile bins fit on a reference distribution."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if not len(ref) or not len(cur):
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(np.mean(ref), np.mean(cur)) else float("inf")
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_pct = np.histogram(ref, bins=edges)[0].astype(float) / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0].astype(float) / len(cur)
    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def build_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, object]:
    missing_ref = sorted(set(FEATURE_COLUMNS) - set(reference.columns))
    missing_cur = sorted(set(FEATURE_COLUMNS) - set(current.columns))
    if missing_ref or missing_cur:
        raise ValueError(f"Feature columns missing; reference={missing_ref}, current={missing_cur}")
    features: dict[str, dict[str, float | str]] = {}
    for name in FEATURE_COLUMNS:
        psi = population_stability_index(reference[name], current[name])
        if not np.isfinite(psi):
            band = "not_available"
        elif psi < 0.10:
            band = "low"
        elif psi < 0.25:
            band = "moderate"
        else:
            band = "high"
        features[name] = {"psi": psi, "drift_band": band}
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "method": "Population Stability Index (PSI), with reference quantile bins",
        "heuristic_bands": {"low": "PSI < 0.10", "moderate": "0.10 <= PSI < 0.25", "high": "PSI >= 0.25"},
        "features": features,
        "interpretation_note": (
            "PSI is a screening signal, not proof of model degradation. Verify data quality and "
            "forecast errors before deciding to retrain."
        ),
    }


def generate_historical_drift_report(train_path: str | Path, output_path: str | Path) -> dict[str, object]:
    train = load_train(train_path)
    feat = build_training_features(train)
    split_date = pd.Timestamp(feat["date"].quantile(0.8))
    reference = feat.loc[feat["date"] <= split_date]
    current = feat.loc[feat["date"] > split_date]
    report = build_drift_report(reference, current)
    report["reference_period"] = [reference["date"].min().date().isoformat(), reference["date"].max().date().isoformat()]
    report["current_period"] = [current["date"].min().date().isoformat(), current["date"].max().date().isoformat()]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--output-path", default=str(REPORTS_DIR / "historical_drift_report.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = generate_historical_drift_report(args.train_path, args.output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
