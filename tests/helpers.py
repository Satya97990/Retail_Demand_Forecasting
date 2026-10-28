from __future__ import annotations

import numpy as np
import pandas as pd


def make_sales_frame(days: int = 45, stores: tuple[int, ...] = (1,), items: tuple[int, ...] = (1,)) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for store in stores:
        for item in items:
            for idx, dt in enumerate(dates):
                rows.append({"date": dt, "store": store, "item": item, "sales": float(10 + idx + store + item)})
    return pd.DataFrame(rows)


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


class LagOnePlusModel:
    def predict(self, features):
        return features["lag_1"].to_numpy(dtype=float) + 1.0
