import pandas as pd

from retail_forecasting.forecast import recursive_forecast
from tests.helpers import ConstantModel, LagOnePlusModel, make_sales_frame


def test_recursive_forecast_appends_predictions_to_history():
    history = make_sales_frame(days=35)
    latest = history["date"].max()
    future = pd.DataFrame(
        {"date": [latest + pd.Timedelta(days=2), latest + pd.Timedelta(days=1)], "store": [1, 1], "item": [1, 1]}
    )
    result = recursive_forecast(history, future, model=LagOnePlusModel())
    # Output remains in input row order even though recursive calculation is chronological.
    assert result["date"].tolist() == future["date"].tolist()
    assert result["sales"].tolist() == [48.0, 47.0]


def test_seasonal_naive_uses_weekly_lag():
    history = make_sales_frame(days=35)
    latest = history["date"].max()
    future = pd.DataFrame({"date": [latest + pd.Timedelta(days=1)], "store": [1], "item": [1]})
    result = recursive_forecast(history, future, strategy="seasonal_naive")
    expected = history.sort_values("date")["sales"].iloc[-7]
    assert result["sales"].iloc[0] == expected


def test_model_forecast_clips_negative_values():
    history = make_sales_frame(days=35)
    latest = history["date"].max()
    future = pd.DataFrame({"date": [latest + pd.Timedelta(days=1)], "store": [1], "item": [1]})
    result = recursive_forecast(history, future, model=ConstantModel(-10))
    assert result["sales"].iloc[0] == 0
