"""Tests for src/serve.py — FastAPI endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from features import add_lag_features
from serve import _metrics, _ModelState, _ServiceMetrics, app
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from train import LSTMAutoencoder

# ── fixtures ──────────────────────────────────────────────────────────────────


N_FEATURES = 25
WINDOW_SIZE = 16  # small for tests


def _make_scaler(n_features: int = N_FEATURES) -> StandardScaler:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((500, n_features)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler


def _make_if_state(n_features: int = N_FEATURES, n_lags: int = 3) -> _ModelState:
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((300, n_features)).astype(np.float32)
    scaler = _make_scaler(n_features)
    flat = add_lag_features(scaler.transform(raw), n_lags)  # (300-n_lags, n_features*(1+n_lags))
    model = IsolationForest(n_estimators=10, random_state=0)
    model.fit(flat)
    return _ModelState(
        model_type="isolation_forest",
        model=model,
        scaler=scaler,
        threshold=0.0,
        alias="champion",
    )


def _make_lstm_state(
    n_features: int = N_FEATURES,
    window_size: int = WINDOW_SIZE,
    threshold: float = 0.05,
) -> _ModelState:
    model = LSTMAutoencoder(n_features=n_features, hidden_size=8, num_layers=1, seq_len=window_size)
    model.eval()
    return _ModelState(
        model_type="lstm_autoencoder",
        model=model,
        scaler=_make_scaler(n_features),
        threshold=threshold,
        alias="champion",
    )


def _make_readings(n: int = 80, n_features: int = N_FEATURES) -> list[list[float]]:
    rng = np.random.default_rng(1)
    return rng.standard_normal((n, n_features)).tolist()


@pytest.fixture()
def if_client(tmp_settings):
    """TestClient with IF model injected, window_size patched small."""
    import serve
    original_state = serve._state
    original_metrics = serve._metrics
    serve._state = _make_if_state()
    serve._metrics = _ServiceMetrics()
    with patch("serve.get_settings", return_value=tmp_settings):
        tmp_settings.n_lag_features = 3
        tmp_settings.if_contamination = 0.05
        yield TestClient(app)
    serve._state = original_state
    serve._metrics = original_metrics


@pytest.fixture()
def lstm_client(tmp_settings):
    """TestClient with LSTM model injected, window_size patched small."""
    import serve
    original_state = serve._state
    original_metrics = serve._metrics
    serve._state = _make_lstm_state(window_size=WINDOW_SIZE)
    serve._metrics = _ServiceMetrics()
    with patch("serve.get_settings", return_value=tmp_settings):
        tmp_settings.window_size = WINDOW_SIZE
        tmp_settings.step_size = 1
        yield TestClient(app)
    serve._state = original_state
    serve._metrics = original_metrics


@pytest.fixture()
def no_model_client():
    """TestClient with no model loaded."""
    import serve
    original_state = serve._state
    serve._state = None
    yield TestClient(app)
    serve._state = original_state


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_ok_with_model(lstm_client) -> None:
    r = lstm_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_type"] == "lstm_autoencoder"
    assert body["model_alias"] == "champion"


def test_health_degraded_without_model(no_model_client) -> None:
    r = no_model_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


def test_health_returns_api_version(lstm_client) -> None:
    r = lstm_client.get("/health")
    assert "api_version" in r.json()


# ── /metrics ──────────────────────────────────────────────────────────────────


def test_metrics_zero_at_start(lstm_client) -> None:
    r = lstm_client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_requests"] == 0
    assert body["anomaly_rate"] == 0.0


def test_metrics_increments_after_predict(lstm_client, tmp_settings) -> None:
    readings = _make_readings(n=WINDOW_SIZE + 5)
    lstm_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    r = lstm_client.get("/metrics")
    assert r.json()["total_requests"] == 1


# ── /predict — LSTM ───────────────────────────────────────────────────────────


def test_predict_lstm_returns_200(lstm_client) -> None:
    readings = _make_readings(n=WINDOW_SIZE + 5)
    r = lstm_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    assert r.status_code == 200


def test_predict_lstm_response_schema(lstm_client) -> None:
    readings = _make_readings(n=WINDOW_SIZE + 5)
    r = lstm_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    body = r.json()
    assert body["channel_id"] == "P-1"
    assert body["model_type"] == "lstm_autoencoder"
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) > 0
    first = body["predictions"][0]
    assert "timestep" in first
    assert "score" in first
    assert "is_anomaly" in first


def test_predict_lstm_too_few_timesteps_returns_422(lstm_client) -> None:
    readings = _make_readings(n=WINDOW_SIZE - 1)
    r = lstm_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    assert r.status_code == 422


def test_predict_lstm_scores_are_non_negative(lstm_client) -> None:
    readings = _make_readings(n=WINDOW_SIZE + 10)
    r = lstm_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    for pred in r.json()["predictions"]:
        assert pred["score"] >= 0.0


# ── /predict — IF ─────────────────────────────────────────────────────────────


def test_predict_if_returns_200(if_client) -> None:
    readings = _make_readings(n=20)
    r = if_client.post("/predict", json={"channel_id": "S-1", "readings": readings})
    assert r.status_code == 200


def test_predict_if_response_schema(if_client) -> None:
    readings = _make_readings(n=20)
    r = if_client.post("/predict", json={"channel_id": "S-1", "readings": readings})
    body = r.json()
    assert body["model_type"] == "isolation_forest"
    assert len(body["predictions"]) > 0


def test_predict_if_too_few_timesteps_returns_422(if_client, tmp_settings) -> None:
    readings = _make_readings(n=tmp_settings.n_lag_features)
    r = if_client.post("/predict", json={"channel_id": "S-1", "readings": readings})
    assert r.status_code == 422


# ── /predict — validation ─────────────────────────────────────────────────────


def test_predict_empty_readings_returns_422(lstm_client) -> None:
    r = lstm_client.post("/predict", json={"channel_id": "P-1", "readings": []})
    assert r.status_code == 422


def test_predict_jagged_readings_returns_422(lstm_client) -> None:
    readings = [[0.1] * N_FEATURES, [0.2] * (N_FEATURES - 1)]
    r = lstm_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    assert r.status_code == 422


def test_predict_503_without_model(no_model_client) -> None:
    readings = _make_readings(n=80)
    r = no_model_client.post("/predict", json={"channel_id": "P-1", "readings": readings})
    assert r.status_code == 503
