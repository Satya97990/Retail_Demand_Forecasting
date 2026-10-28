"""Generate compact exploratory and model-evaluation figures for the project report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from retail_forecasting.config import FIGURES_DIR, REPORTS_DIR


def create_project_figures(
    train_df: pd.DataFrame,
    validation_predictions_path: str | Path,
    metrics_path: str | Path,
    figures_dir: str | Path = FIGURES_DIR,
) -> list[str]:
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    daily = train_df.groupby("date", as_index=False)["sales"].sum().sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily["date"], daily["sales"], linewidth=0.8)
    ax.set(title="Total daily sales across all stores and items", xlabel="Date", ylabel="Units sold")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = figures / "daily_total_sales.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    monthly = train_df.assign(month=pd.to_datetime(train_df["date"]).dt.month).groupby("month")["sales"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(monthly.index.astype(str), monthly.values)
    ax.set(title="Average daily item sales by calendar month", xlabel="Month", ylabel="Mean units sold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures / "monthly_seasonality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(str(path))

    val_path = Path(validation_predictions_path)
    metric_path = Path(metrics_path)
    if val_path.exists() and metric_path.exists():
        val = pd.read_csv(val_path, parse_dates=["date"])
        val_daily = val.groupby("date", as_index=False).agg(
            actual_sales=("actual_sales", "sum"),
            model_prediction=("model_prediction", "sum"),
            seasonal_naive_prediction=("seasonal_naive_prediction", "sum"),
        )
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(val_daily["date"], val_daily["actual_sales"], label="Actual", linewidth=1.8)
        ax.plot(val_daily["date"], val_daily["model_prediction"], label="ML model", linewidth=1.3)
        ax.plot(val_daily["date"], val_daily["seasonal_naive_prediction"], label="Seasonal-naive", linewidth=1.0)
        ax.set(title="90-day recursive validation: total daily sales", xlabel="Date", ylabel="Units sold")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = figures / "validation_forecasts.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        model_metrics = metrics["validation_model_metrics"]
        baseline_metrics = metrics["seasonal_naive_metrics"]
        labels = ["MAE", "RMSE", "WAPE (%)", "sMAPE (%)"]
        keys = ["mae", "rmse", "wape_pct", "smape_pct"]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        x = range(len(labels))
        width = 0.36
        ax.bar([i - width / 2 for i in x], [model_metrics[k] for k in keys], width, label="ML model")
        ax.bar([i + width / 2 for i in x], [baseline_metrics[k] for k in keys], width, label="Seasonal-naive")
        ax.set_xticks(list(x), labels)
        ax.set_title("Model versus seasonal-naive baseline on the same 90-day holdout")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = figures / "model_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))
    return saved
