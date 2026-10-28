"""FastAPI service for iterative daily store-item demand forecasts."""
from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from retail_forecasting.config import DEFAULT_MODEL_PATH, FEATURE_COLUMNS
from retail_forecasting.features import check_daily_continuity, make_feature_row
from retail_forecasting.modeling import load_model

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
app = FastAPI(
    title="Retail Demand Forecasting API",
    version="1.0.0",
    description=(
        "Generate recursive daily demand forecasts for a store/item pair. "
        "Supply at least 28 consecutive days of historical sales."
    ),
)


class HistoryPoint(BaseModel):
    date: date
    sales: float = Field(ge=0, description="Observed sales on that date")


class ForecastRequest(BaseModel):
    store: int = Field(ge=1, description="Store identifier")
    item: int = Field(ge=1, description="Item identifier")
    history: list[HistoryPoint] = Field(min_length=28, description="Chronological daily sales history")
    horizon_days: Annotated[int, Field(ge=1, le=90)] = 30

    @model_validator(mode="after")
    def validate_history(self) -> "ForecastRequest":
        dates = [point.date for point in self.history]
        if not check_daily_continuity(dates):
            raise ValueError("history must contain unique, sorted, consecutive daily dates")
        return self


class ForecastPoint(BaseModel):
    date: date
    sales: float = Field(ge=0)


class ForecastResponse(BaseModel):
    store: int
    item: int
    horizon_days: int
    forecasts: list[ForecastPoint]


@lru_cache(maxsize=1)
def _get_model():
    return load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, object]:
    artifact_exists = MODEL_PATH.exists()
    return {
        "status": "ready" if artifact_exists else "model_artifact_missing",
        "model_artifact_exists": artifact_exists,
        "model_path": str(MODEL_PATH),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    try:
        model = _get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not load model artifact: {exc}") from exc

    history_points = request.history
    history_values = [float(point.sales) for point in history_points]
    latest_date = history_points[-1].date
    forecast_points: list[ForecastPoint] = []
    for step in range(1, request.horizon_days + 1):
        target_date = latest_date + timedelta(days=step)
        row = make_feature_row(target_date, request.store, request.item, history_values)
        matrix = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        prediction = float(np.clip(model.predict(matrix)[0], 0.0, None))
        forecast_points.append(ForecastPoint(date=target_date, sales=prediction))
        history_values.append(prediction)
    return ForecastResponse(
        store=request.store,
        item=request.item,
        horizon_days=request.horizon_days,
        forecasts=forecast_points,
    )
