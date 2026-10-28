"""Recursive forecasting so later forecast days use prior predictions, not actual future sales."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from retail_forecasting.config import FEATURE_COLUMNS, KEY_COLUMNS
from retail_forecasting.features import make_feature_row


def recursive_forecast(
    history: pd.DataFrame,
    future: pd.DataFrame,
    model: Any | None = None,
    strategy: str = "model",
    seasonal_period: int = 7,
) -> pd.DataFrame:
    """Forecast future rows in date order and return input columns plus `sales` predictions.

    `history` must have date/store/item/sales. `future` must have date/store/item and may
    include id. Predictions are generated one date at a time; for horizons beyond one week,
    predicted values are appended to history before forecasting later dates.
    """
    if strategy not in {"model", "seasonal_naive"}:
        raise ValueError("strategy must be 'model' or 'seasonal_naive'")
    if strategy == "model" and model is None:
        raise ValueError("A trained model must be supplied when strategy='model'.")
    required_history = {"date", "store", "item", "sales"}
    required_future = {"date", "store", "item"}
    if missing := required_history - set(history.columns):
        raise ValueError(f"History is missing columns: {sorted(missing)}")
    if missing := required_future - set(future.columns):
        raise ValueError(f"Future data is missing columns: {sorted(missing)}")
    if future.empty:
        result = future.copy()
        result["sales"] = pd.Series(dtype=float)
        return result

    hist = history[["date", "store", "item", "sales"]].copy()
    fut = future.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    fut["date"] = pd.to_datetime(fut["date"])
    hist = hist.sort_values([*KEY_COLUMNS, "date"])
    fut["original_order"] = np.arange(len(fut))
    fut = fut.sort_values(["date", *KEY_COLUMNS])

    histories: dict[tuple[int, int], list[float]] = defaultdict(list)
    last_dates: dict[tuple[int, int], pd.Timestamp] = {}
    for row in hist.itertuples(index=False):
        key = (int(row.store), int(row.item))
        histories[key].append(float(row.sales))
        last_dates[key] = pd.Timestamp(row.date)

    predictions: list[dict[str, float | int | pd.Timestamp]] = []
    for target_date, day_rows in fut.groupby("date", sort=True):
        feature_rows: list[dict[str, float | int]] = []
        day_keys: list[tuple[int, int]] = []
        for row in day_rows.itertuples(index=False):
            key = (int(row.store), int(row.item))
            values = histories[key]
            if len(values) < 28:
                raise ValueError(f"At least 28 history values required for store/item {key}.")
            if key in last_dates and pd.Timestamp(target_date) != last_dates[key] + pd.Timedelta(days=1):
                raise ValueError(
                    f"Non-consecutive forecast date for store/item {key}: "
                    f"last history date {last_dates[key].date()}, target {pd.Timestamp(target_date).date()}"
                )
            day_keys.append(key)
            if strategy == "model":
                feature_rows.append(make_feature_row(target_date, key[0], key[1], values))
            else:
                feature_rows.append({"seasonal_naive_prediction": float(values[-seasonal_period])})
        if strategy == "model":
            matrix = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
            day_predictions = np.asarray(model.predict(matrix), dtype=float)
        else:
            day_predictions = np.asarray(
                [entry["seasonal_naive_prediction"] for entry in feature_rows], dtype=float
            )
        day_predictions = np.clip(day_predictions, 0.0, None)
        for row, key, pred in zip(day_rows.itertuples(index=False), day_keys, day_predictions):
            predictions.append(
                {
                    "original_order": int(row._asdict()["original_order"]),
                    "id": getattr(row, "id", None),
                    "date": pd.Timestamp(target_date),
                    "store": key[0],
                    "item": key[1],
                    "sales": float(pred),
                }
            )
            histories[key].append(float(pred))
            last_dates[key] = pd.Timestamp(target_date)

    pred_df = pd.DataFrame(predictions).sort_values("original_order")
    result = future.copy().reset_index(drop=True)
    result["sales"] = pred_df["sales"].to_numpy()
    return result.drop(columns=["original_order"], errors="ignore")
