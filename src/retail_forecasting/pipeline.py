"""One-command end-to-end local pipeline: validate, train/evaluate, predict, report."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from retail_forecasting.analysis import create_project_figures
from retail_forecasting.config import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from retail_forecasting.data import load_test, load_train, validate_dataset_coverage
from retail_forecasting.monitoring import generate_historical_drift_report
from retail_forecasting.modeling import train_and_evaluate
from retail_forecasting.predict import create_submission


def run_pipeline(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    use_mlflow: bool = True,
) -> dict[str, object]:
    logging.info("Step 1/5: validate data contracts and coverage")
    train, test = load_train(train_path), load_test(test_path)
    manifest = validate_dataset_coverage(train, test)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dataset_profile.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Step 2/5: train, perform 90-day recursive validation, save model")
    training_result = train_and_evaluate(
        train,
        model_dir=MODEL_DIR,
        reports_dir=REPORTS_DIR,
        processed_dir=PROCESSED_DATA_DIR,
        use_mlflow=use_mlflow,
    )

    logging.info("Step 3/5: generate recursive test forecasts")
    submission_path = PROCESSED_DATA_DIR / "submission.csv"
    submission = create_submission(train_path, test_path, MODEL_DIR / "model.joblib", submission_path)

    logging.info("Step 4/5: create historical drift screening report")
    drift = generate_historical_drift_report(train_path, REPORTS_DIR / "historical_drift_report.json")

    logging.info("Step 5/5: create plots for the project report")
    figures = create_project_figures(
        train,
        training_result["validation_predictions_path"],
        training_result["validation_metrics_path"],
    )
    return {
        "dataset_profile": manifest,
        "model_validation_metrics": training_result["model_metrics"],
        "seasonal_naive_metrics": training_result["seasonal_naive_metrics"],
        "model_path": training_result["model_path"],
        "submission_path": str(submission_path),
        "submission_rows": len(submission),
        "drift_report_path": str(REPORTS_DIR / "historical_drift_report.json"),
        "figures": figures,
        "drift_feature_count": len(drift["features"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_pipeline(args.train_path, args.test_path, use_mlflow=not args.no_mlflow)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
