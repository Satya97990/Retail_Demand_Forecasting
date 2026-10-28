"""Loading and validation for the Kaggle Store Item Demand Forecasting data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_forecasting.config import TEST_COLUMNS, TRAIN_COLUMNS


class DataValidationError(ValueError):
    """Raised when an input dataset does not meet the expected contract."""


def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Download the competition files from "
            "https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data "
            "and place them in data/raw/. See data/README.md."
        )
    return pd.read_csv(path)


def validate_train_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the training dataframe without silently fixing bad rows."""
    missing = sorted(set(TRAIN_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Training data is missing required columns: {missing}")
    out = df[TRAIN_COLUMNS].copy()
    if out.isna().any().any():
        counts = out.isna().sum()
        bad = counts[counts > 0].to_dict()
        raise DataValidationError(f"Training data contains missing values: {bad}")
    try:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        out["store"] = pd.to_numeric(out["store"], errors="raise").astype("int64")
        out["item"] = pd.to_numeric(out["item"], errors="raise").astype("int64")
        out["sales"] = pd.to_numeric(out["sales"], errors="raise").astype("float64")
    except Exception as exc:
        raise DataValidationError(f"Could not parse training columns: {exc}") from exc
    if (out["sales"] < 0).any():
        raise DataValidationError("Training target 'sales' must be non-negative.")
    if (out["store"] <= 0).any() or (out["item"] <= 0).any():
        raise DataValidationError("Store and item identifiers must be positive integers.")
    duplicates = out.duplicated(["date", "store", "item"], keep=False)
    if duplicates.any():
        example = out.loc[duplicates, ["date", "store", "item"]].head(3).to_dict("records")
        raise DataValidationError(
            "Duplicate observations for the same date/store/item were found; "
            f"investigate before training. Examples: {example}"
        )
    return out.sort_values(["date", "store", "item"]).reset_index(drop=True)


def validate_test_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the competition test frame."""
    missing = sorted(set(TEST_COLUMNS) - set(df.columns))
    if missing:
        raise DataValidationError(f"Test data is missing required columns: {missing}")
    out = df[TEST_COLUMNS].copy()
    if out.isna().any().any():
        bad = out.isna().sum()
        raise DataValidationError(f"Test data contains missing values: {bad[bad > 0].to_dict()}")
    try:
        out["id"] = pd.to_numeric(out["id"], errors="raise").astype("int64")
        out["date"] = pd.to_datetime(out["date"], errors="raise")
        out["store"] = pd.to_numeric(out["store"], errors="raise").astype("int64")
        out["item"] = pd.to_numeric(out["item"], errors="raise").astype("int64")
    except Exception as exc:
        raise DataValidationError(f"Could not parse test columns: {exc}") from exc
    if out["id"].duplicated().any():
        raise DataValidationError("Test data has duplicate IDs.")
    if out.duplicated(["date", "store", "item"]).any():
        raise DataValidationError("Test data has duplicate date/store/item observations.")
    if (out["store"] <= 0).any() or (out["item"] <= 0).any():
        raise DataValidationError("Store and item identifiers must be positive integers.")
    return out.sort_values(["date", "store", "item"]).reset_index(drop=True)


def load_train(path: str | Path) -> pd.DataFrame:
    return validate_train_frame(_read_csv(path))


def load_test(path: str | Path) -> pd.DataFrame:
    return validate_test_frame(_read_csv(path))


def validate_dataset_coverage(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    """Create a manifest and verify test item-store pairs exist in training history."""
    train_pairs = set(zip(train["store"].tolist(), train["item"].tolist()))
    test_pairs = set(zip(test["store"].tolist(), test["item"].tolist()))
    unseen = sorted(test_pairs - train_pairs)
    if unseen:
        raise DataValidationError(
            f"Test set contains item/store combinations absent from training: {unseen[:10]}"
        )
    if test["date"].min() <= train["date"].max():
        raise DataValidationError(
            "Forecast test dates must begin after the final observed training date; "
            "overlapping dates could create leakage."
        )
    expected_days = int((train["date"].max() - train["date"].min()).days + 1)
    observed_days = train.groupby(["store", "item"])["date"].nunique()
    incomplete = observed_days[observed_days != expected_days]
    if not incomplete.empty:
        examples = {str(k): int(v) for k, v in incomplete.head(10).items()}
        raise DataValidationError(
            "Some store/item series do not have one observation per calendar day across the "
            f"training range ({expected_days} days expected). Examples: {examples}"
        )
    return {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "stores": int(train["store"].nunique()),
        "items": int(train["item"].nunique()),
        "store_item_pairs": int(train[["store", "item"]].drop_duplicates().shape[0]),
        "unseen_test_pairs": 0,
    }
