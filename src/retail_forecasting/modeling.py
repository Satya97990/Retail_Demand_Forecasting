"""Model training, chronological validation, artifact persistence, and optional MLflow tracking."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from retail_forecasting.config import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
)
from retail_forecasting.features import build_training_features
from retail_forecasting.forecast import recursive_forecast
from retail_forecasting.metrics import calculate_metrics

LOGGER = logging.getLogger(__name__)


def make_model(random_state: int = 42) -> HistGradientBoostingRegressor:
    """Return the project's baseline tree-boosting model with deterministic settings."""
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=random_state,
    )


def _fit_model(model: Any, features: pd.DataFrame) -> None:
    model.fit(features[FEATURE_COLUMNS], features["sales"])


def train_and_evaluate(
    train_df: pd.DataFrame,
    model_dir: str | Path = MODEL_DIR,
    reports_dir: str | Path = REPORTS_DIR,
    processed_dir: str | Path = PROCESSED_DATA_DIR,
    use_mlflow: bool = True,
    register_model: bool = False,
    model_name: str = "retail-demand-forecasting",
) -> dict[str, Any]:
    """Run a 90-day recursive validation, compare with seasonal-naive, fit final model, save.

    The external chronological validation is not randomly shuffled. Forecasts for future days
    use prior predictions recursively rather than actual sales from the holdout window.
    """
    if train_df["date"].nunique() < 120:
        raise ValueError("At least 120 unique daily dates are recommended for temporal validation.")
    model_dir = Path(model_dir)
    reports_dir = Path(reports_dir)
    processed_dir = Path(processed_dir)
    for directory in (model_dir, reports_dir, processed_dir):
        directory.mkdir(parents=True, exist_ok=True)

    max_date = pd.Timestamp(train_df["date"].max())
    validation_start = max_date - pd.Timedelta(days=89)
    history = train_df.loc[train_df["date"] < validation_start].copy()
    validation_actual = train_df.loc[train_df["date"] >= validation_start].copy()
    if history.empty or validation_actual.empty:
        raise ValueError("Chronological split produced an empty history or validation window.")
    if validation_actual["date"].nunique() != 90:
        LOGGER.warning(
            "Validation window has %d unique dates rather than 90; the dataset may have gaps.",
            validation_actual["date"].nunique(),
        )

    LOGGER.info("Building leakage-safe lag and rolling features for %s training rows", len(train_df))
    all_features = build_training_features(train_df)
    historical_features = all_features.loc[all_features["date"] < validation_start]
    if historical_features.empty:
        raise ValueError("No training rows remain after feature warm-up and temporal split.")

    validation_future = validation_actual[["date", "store", "item"]].copy()
    LOGGER.info(
        "Training validation model on %s rows; recursively forecasting %s validation rows",
        len(historical_features), len(validation_future),
    )
    validation_model = make_model()
    _fit_model(validation_model, historical_features)
    model_forecast = recursive_forecast(history, validation_future, model=validation_model)
    baseline_forecast = recursive_forecast(
        history, validation_future, strategy="seasonal_naive", seasonal_period=7
    )

    actual = validation_actual[["date", "store", "item", "sales"]].rename(
        columns={"sales": "actual_sales"}
    )
    model_eval = model_forecast[["date", "store", "item", "sales"]].rename(
        columns={"sales": "model_prediction"}
    )
    baseline_eval = baseline_forecast[["date", "store", "item", "sales"]].rename(
        columns={"sales": "seasonal_naive_prediction"}
    )
    validation_results = actual.merge(model_eval, on=["date", "store", "item"], validate="one_to_one")
    validation_results = validation_results.merge(
        baseline_eval, on=["date", "store", "item"], validate="one_to_one"
    )
    model_metrics = calculate_metrics(
        validation_results["actual_sales"], validation_results["model_prediction"]
    )
    baseline_metrics = calculate_metrics(
        validation_results["actual_sales"], validation_results["seasonal_naive_prediction"]
    )
    validation_results.to_csv(processed_dir / "validation_predictions.csv", index=False)

    metadata: dict[str, Any] = {
        "project": "Retail Demand Forecasting",
        "model_type": "HistGradientBoostingRegressor",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "sales",
        "forecasting_strategy": "recursive multi-step daily forecasting",
        "validation_strategy": "last 90 days held out; no random shuffling",
        "validation_start": validation_start.date().isoformat(),
        "validation_end": max_date.date().isoformat(),
        "validation_horizon_days": int(validation_actual["date"].nunique()),
        "validation_rows": int(len(validation_actual)),
        "validation_model_metrics": model_metrics,
        "seasonal_naive_metrics": baseline_metrics,
        "training_rows_after_feature_warmup": int(len(all_features)),
        "final_training_start": pd.Timestamp(train_df["date"].min()).date().isoformat(),
        "final_training_end": max_date.date().isoformat(),
        "stores": int(train_df["store"].nunique()),
        "items": int(train_df["item"].nunique()),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_state": 42,
        "notes": [
            "Validation forecasts are recursive: later validation days use earlier predictions.",
            "The final model is refit on all available historical training rows after validation.",
            "Kaggle competition data is excluded from Git using .gitignore.",
        ],
    }
    metrics_path = reports_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info("Fitting final model on all %s eligible historical feature rows", len(all_features))
    final_model = make_model()
    _fit_model(final_model, all_features)
    model_path = model_dir / "model.joblib"
    metadata_path = model_dir / "metadata.json"
    joblib.dump(final_model, model_path, compress=3)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if use_mlflow:
        try:
            import mlflow
            import mlflow.sklearn

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                local_tracking = (model_dir.parent / "mlruns").resolve()
                mlflow.set_tracking_uri(local_tracking.as_uri())
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "retail-demand-forecasting"))
            with mlflow.start_run(run_name="hist-gradient-boosting-90-day-validation"):
                mlflow.log_params(
                    {
                        "model_type": "HistGradientBoostingRegressor",
                        "learning_rate": 0.08,
                        "max_iter": 100,
                        "max_leaf_nodes": 31,
                        "min_samples_leaf": 30,
                        "l2_regularization": 1.0,
                        "validation_horizon_days": metadata["validation_horizon_days"],
                        "feature_count": len(FEATURE_COLUMNS),
                    }
                )
                for key, value in model_metrics.items():
                    mlflow.log_metric(f"model_{key}", value)
                for key, value in baseline_metrics.items():
                    mlflow.log_metric(f"seasonal_naive_{key}", value)
                mlflow.log_param("final_training_rows", len(all_features))
                mlflow.log_artifact(str(metrics_path), artifact_path="reports")
                mlflow.sklearn.log_model(final_model, artifact_path="model")
                if register_model:
                    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
                    mlflow.register_model(model_uri, model_name)
            LOGGER.info("MLflow tracking completed.")
        except ImportError:
            LOGGER.warning("MLflow is not installed; model training succeeded without experiment tracking.")
        except Exception as exc:  # MLflow server/registry is optional for local reproducibility.
            LOGGER.exception("MLflow tracking failed, but the local model artifacts were saved: %s", exc)
            if register_model:
                raise

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "validation_metrics_path": str(metrics_path),
        "validation_predictions_path": str(processed_dir / "validation_predictions.csv"),
        "model_metrics": model_metrics,
        "seasonal_naive_metrics": baseline_metrics,
        "metadata": metadata,
    }


def load_model(model_path: str | Path = MODEL_DIR / "model.joblib") -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {path}. Run `python -m retail_forecasting.pipeline` first."
        )
    return joblib.load(path)
