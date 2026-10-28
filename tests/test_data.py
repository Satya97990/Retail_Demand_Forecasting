import pandas as pd
import pytest

from retail_forecasting.data import DataValidationError, validate_test_frame, validate_train_frame


def test_training_frame_is_parsed_and_sorted():
    df = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "store": [1, 1],
            "item": [1, 1],
            "sales": [12, 10],
        }
    )
    result = validate_train_frame(df)
    assert result["date"].is_monotonic_increasing
    assert result["sales"].tolist() == [10.0, 12.0]


def test_training_data_rejects_duplicate_date_store_item():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "store": [1, 1],
            "item": [2, 2],
            "sales": [10, 13],
        }
    )
    with pytest.raises(DataValidationError, match="Duplicate observations"):
        validate_train_frame(df)


def test_training_data_rejects_negative_sales():
    df = pd.DataFrame({"date": ["2024-01-01"], "store": [1], "item": [1], "sales": [-1]})
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_train_frame(df)


def test_test_data_rejects_duplicate_ids():
    df = pd.DataFrame(
        {"id": [1, 1], "date": ["2024-01-01", "2024-01-02"], "store": [1, 1], "item": [1, 1]}
    )
    with pytest.raises(DataValidationError, match="duplicate IDs"):
        validate_test_frame(df)


def test_dataset_coverage_accepts_future_test_dates_and_complete_series():
    from retail_forecasting.data import validate_dataset_coverage

    train = validate_train_frame(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=30).astype(str),
                "store": [1] * 30,
                "item": [1] * 30,
                "sales": list(range(30)),
            }
        )
    )
    test = validate_test_frame(
        pd.DataFrame({"id": [1, 2], "date": ["2024-01-31", "2024-02-01"], "store": [1, 1], "item": [1, 1]})
    )
    profile = validate_dataset_coverage(train, test)
    assert profile["train_rows"] == 30
    assert profile["unseen_test_pairs"] == 0


def test_dataset_coverage_rejects_missing_daily_observations():
    from retail_forecasting.data import validate_dataset_coverage

    train = validate_train_frame(
        pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-03"],
                "store": [1, 1],
                "item": [1, 1],
                "sales": [10, 12],
            }
        )
    )
    test = validate_test_frame(
        pd.DataFrame({"id": [1], "date": ["2024-01-04"], "store": [1], "item": [1]})
    )
    with pytest.raises(DataValidationError, match="one observation per calendar day"):
        validate_dataset_coverage(train, test)
