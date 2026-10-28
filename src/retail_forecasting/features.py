"""Leakage-aware calendar, lag, and rolling-window feature generation."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from retail_forecasting.config import FEATURE_COLUMNS, KEY_COLUMNS


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt = pd.to_datetime(out["date"])
    out["day_of_week"] = dt.dt.dayofweek.astype("int16")
    out["day_of_month"] = dt.dt.day.astype("int16")
    out["month"] = dt.dt.month.astype("int16")
    out["day_of_year"] = dt.dt.dayofyear.astype("int16")
    out["week_of_year"] = dt.dt.isocalendar().week.astype("int16")
    out["quarter"] = dt.dt.quarter.astype("int8")
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    return out


def build_training_features(train: pd.DataFrame) -> pd.DataFrame:
    """Build row-wise training features using only observations strictly before each row.

    The lag and rolling features use shift(1), preventing current target sales from entering
    the feature values. Initial rows without enough history are dropped intentionally.
    """
    df = train.sort_values([*KEY_COLUMNS, "date"]).reset_index(drop=True).copy()
    df = add_calendar_features(df)
    grouped = df.groupby(KEY_COLUMNS, sort=False)["sales"]
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 14, 28):
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda values, w=window: values.shift(1).rolling(window=w, min_periods=w).mean()
        )
    df["rolling_std_7"] = grouped.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=7).std(ddof=0)
    )
    required = FEATURE_COLUMNS + ["sales", "date"]
    return df.dropna(subset=required).reset_index(drop=True)


def make_feature_row(
    target_date: date | pd.Timestamp | str,
    store: int,
    item: int,
    history_values: Sequence[float],
) -> dict[str, float | int]:
    """Create a feature row from sales history before target_date.

    `history_values` must be chronological, with one value per consecutive day and must
    not include the target date or any later observation.
    """
    if len(history_values) < 28:
        raise ValueError("At least 28 prior daily sales observations are required.")
    dt = pd.Timestamp(target_date)
    values = np.asarray(history_values, dtype=float)
    last_7 = values[-7:]
    last_14 = values[-14:]
    last_28 = values[-28:]
    return {
        "store": int(store),
        "item": int(item),
        "day_of_week": int(dt.dayofweek),
        "day_of_month": int(dt.day),
        "month": int(dt.month),
        "day_of_year": int(dt.dayofyear),
        "week_of_year": int(dt.isocalendar().week),
        "quarter": int(dt.quarter),
        "is_weekend": int(dt.dayofweek >= 5),
        "lag_1": float(values[-1]),
        "lag_7": float(values[-7]),
        "lag_14": float(values[-14]),
        "lag_28": float(values[-28]),
        "rolling_mean_7": float(last_7.mean()),
        "rolling_mean_14": float(last_14.mean()),
        "rolling_mean_28": float(last_28.mean()),
        "rolling_std_7": float(last_7.std(ddof=0)),
    }


def check_daily_continuity(dates: Sequence[date | pd.Timestamp | str]) -> bool:
    """Return true when dates are unique, sorted, and consecutive daily observations."""
    dt = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    return bool(
        len(dt) > 0
        and dt.is_unique
        and dt.is_monotonic_increasing
        and (len(dt) < 2 or (dt[1:] - dt[:-1] == pd.Timedelta(days=1)).all())
    )
