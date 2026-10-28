from datetime import date, timedelta

from fastapi.testclient import TestClient

import app.main as api
from tests.helpers import ConstantModel


def test_forecast_api_returns_requested_horizon(monkeypatch):
    monkeypatch.setattr(api, "_get_model", lambda: ConstantModel(42.5))
    client = TestClient(api.app)
    first = date(2024, 1, 1)
    history = [
        {"date": (first + timedelta(days=i)).isoformat(), "sales": float(30 + i)}
        for i in range(35)
    ]
    response = client.post(
        "/forecast",
        json={"store": 1, "item": 2, "horizon_days": 3, "history": history},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["forecasts"]) == 3
    assert body["forecasts"][0]["sales"] == 42.5
    assert body["forecasts"][0]["date"] == (first + timedelta(days=35)).isoformat()


def test_forecast_api_rejects_gapped_history():
    client = TestClient(api.app)
    first = date(2024, 1, 1)
    history = [
        {"date": (first + timedelta(days=i + (1 if i >= 10 else 0))).isoformat(), "sales": float(i)}
        for i in range(35)
    ]
    response = client.post(
        "/forecast", json={"store": 1, "item": 2, "horizon_days": 1, "history": history}
    )
    assert response.status_code == 422
