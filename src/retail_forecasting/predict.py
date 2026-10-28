"""Generate recursive test forecasts and an output file matching Kaggle's submission schema."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from retail_forecasting.config import DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, MODEL_DIR, PROCESSED_DATA_DIR
from retail_forecasting.data import load_test, load_train
from retail_forecasting.forecast import recursive_forecast
from retail_forecasting.modeling import load_model


def create_submission(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    model_path: str | Path = MODEL_DIR / "model.joblib",
    output_path: str | Path = PROCESSED_DATA_DIR / "submission.csv",
) -> pd.DataFrame:
    train = load_train(train_path)
    test = load_test(test_path)
    model = load_model(model_path)
    forecast = recursive_forecast(train, test, model=model)
    submission = forecast[["id", "sales"]].copy()
    submission["id"] = submission["id"].astype("int64")
    submission = submission.sort_values("id").reset_index(drop=True)
    if len(submission) != len(test) or submission["id"].duplicated().any():
        raise RuntimeError("Submission row count or ID uniqueness check failed.")
    if submission["sales"].isna().any() or (submission["sales"] < 0).any():
        raise RuntimeError("Submission contains missing or negative predictions.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False, float_format="%.6f")
    logging.info("Saved %d forecasts to %s", len(submission), output)
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--model-path", default=str(MODEL_DIR / "model.joblib"))
    parser.add_argument("--output-path", default=str(PROCESSED_DATA_DIR / "submission.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    submission = create_submission(args.train_path, args.test_path, args.model_path, args.output_path)
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
