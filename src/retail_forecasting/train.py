"""CLI entry point for training and chronological model evaluation."""
from __future__ import annotations

import argparse
import json
import logging

from retail_forecasting.config import DEFAULT_TRAIN_PATH, MODEL_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from retail_forecasting.data import load_train
from retail_forecasting.modeling import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    parser.add_argument("--processed-dir", default=str(PROCESSED_DATA_DIR))
    parser.add_argument("--no-mlflow", action="store_true", help="Skip experiment tracking")
    parser.add_argument("--register-model", action="store_true", help="Register model in MLflow registry")
    parser.add_argument("--model-name", default="retail-demand-forecasting")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train = load_train(args.train_path)
    result = train_and_evaluate(
        train,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
        processed_dir=args.processed_dir,
        use_mlflow=not args.no_mlflow,
        register_model=args.register_model,
        model_name=args.model_name,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "metadata"}, indent=2))
    print("\nModel validation metrics:")
    print(json.dumps(result["model_metrics"], indent=2))
    print("Seasonal-naive baseline metrics:")
    print(json.dumps(result["seasonal_naive_metrics"], indent=2))


if __name__ == "__main__":
    main()
