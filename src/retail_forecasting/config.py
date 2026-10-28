"""Shared project paths and feature definitions."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_TRAIN_PATH = RAW_DATA_DIR / "train.csv"
DEFAULT_TEST_PATH = RAW_DATA_DIR / "test.csv"
DEFAULT_SAMPLE_SUBMISSION_PATH = RAW_DATA_DIR / "sample_submission.csv"
DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
DEFAULT_METADATA_PATH = MODEL_DIR / "metadata.json"

KEY_COLUMNS = ["store", "item"]
TRAIN_COLUMNS = ["date", "store", "item", "sales"]
TEST_COLUMNS = ["id", "date", "store", "item"]
FEATURE_COLUMNS = [
    "store",
    "item",
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "week_of_year",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
]
