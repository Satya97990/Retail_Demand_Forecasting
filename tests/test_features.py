import pandas as pd

from retail_forecasting.features import build_training_features, check_daily_continuity, make_feature_row
from tests.helpers import make_sales_frame


def test_lag_features_only_use_prior_sales():
    df = make_sales_frame(days=40)
    features = build_training_features(df)
    target_date = pd.Timestamp("2020-02-09")
    row = features.loc[features["date"] == target_date].iloc[0]
    prior = df.loc[df["date"] < target_date].sort_values("date")["sales"].tolist()
    assert row["lag_1"] == prior[-1]
    assert row["lag_7"] == prior[-7]
    assert row["rolling_mean_7"] == sum(prior[-7:]) / 7


def test_changing_current_target_does_not_change_its_features():
    df = make_sales_frame(days=40)
    before = build_training_features(df)
    target_date = pd.Timestamp("2020-02-09")
    df.loc[df["date"] == target_date, "sales"] += 1000
    after = build_training_features(df)
    fcols = ["lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_14", "rolling_mean_28"]
    left = before.loc[before["date"] == target_date, fcols].iloc[0]
    right = after.loc[after["date"] == target_date, fcols].iloc[0]
    assert left.equals(right)


def test_make_feature_row_uses_recent_history():
    values = list(range(1, 29))
    row = make_feature_row("2024-03-01", 2, 3, values)
    assert row["lag_1"] == 28
    assert row["lag_7"] == 22
    assert row["rolling_mean_7"] == 25


def test_daily_continuity():
    assert check_daily_continuity(pd.date_range("2024-01-01", periods=4))
    assert not check_daily_continuity(["2024-01-01", "2024-01-03"])
