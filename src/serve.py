"""FastAPI serving layer for SensorFlow anomaly detection.

Endpoints
---------
POST /predict   — score raw sensor readings; returns anomaly scores + flags
GET  /health    — liveness check with model metadata
GET  /metrics   — aggregate prediction statistics since startup

Model loading
-------------
At startup the champion model is loaded from the local ``models/`` directory.
If MLflow is reachable the champion alias is resolved from the registry first;
otherwise the local artefacts are used directly so the service starts offline.

Prediction alignment
--------------------
- Isolation Forest: one score per timestep starting at index ``n_lag_features``
  (earlier timesteps lack full lag context).
- LSTM Autoencoder: one score per window; each score is assigned to the *last*
  timestep of its window, so the first ``window_size - 1`` timesteps are skipped.

The response always includes the timestep index so callers know exactly which
readings each score corresponds to.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import StandardScaler

from config import Settings, get_settings
from evaluate import score_lstm
from features import add_lag_features, sliding_window
from train import LSTMAutoencoder

log = logging.getLogger(__name__)


# ── in-memory state ───────────────────────────────────────────────────────────


@dataclass
class _ModelState:
    model_type: str          # "isolation_forest" | "lstm_autoencoder"
    model: Any               # IsolationForest | LSTMAutoencoder
    scaler: StandardScaler
    threshold: float
    alias: str
    loaded_at: float = field(default_factory=time.time)


@dataclass
class _ServiceMetrics:
    total_requests: int = 0
    total_anomalies: int = 0
    score_sum: float = 0.0
    started_at: float = field(default_factory=time.time)


_state: _ModelState | None = None
_metrics: _ServiceMetrics = _ServiceMetrics()


# ── model loading ─────────────────────────────────────────────────────────────


def _load_model_state(cfg: Settings) -> _ModelState:
    """Load the champion model and scaler from disk.

    Tries to resolve the champion alias from MLflow first; falls back to
    loading whichever local artefact is present if MLflow is unreachable.

    Returns:
        Populated :class:`_ModelState`.

    Raises:
        RuntimeError: If no trained model is found locally.
    """
    scaler_path = cfg.models_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise RuntimeError(f"Scaler not found at {scaler_path}. Run features.py first.")
    scaler: StandardScaler = joblib.load(scaler_path)

    # Prefer the LSTM if both artefacts exist — it typically has higher F1.
    lstm_path = cfg.models_dir / "lstm_autoencoder.pt"
    if_path = cfg.models_dir / "isolation_forest.joblib"

    # Try to discover the real champion from MLflow.
    champion_type = _resolve_champion_type(cfg)

    if champion_type == "lstm_autoencoder" and lstm_path.exists():
        checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=True)
        n_features = int(scaler.n_features_in_)
        model = LSTMAutoencoder(
            n_features=n_features,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            seq_len=cfg.window_size,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        threshold = float(checkpoint["threshold"])
        log.info("loaded LSTM Autoencoder (threshold=%.5f)", threshold)
        return _ModelState("lstm_autoencoder", model, scaler, threshold, cfg.mlflow_champion_alias)

    if if_path.exists():
        import joblib as jl
        if_model = jl.load(if_path)
        # Derive threshold: top contamination-fraction of scores = anomalous.
        # We store 0.0 as a sentinel; _score_if computes it dynamically.
        log.info("loaded Isolation Forest")
        return _ModelState("isolation_forest", if_model, scaler, 0.0, cfg.mlflow_champion_alias)

    raise RuntimeError(
        f"No trained model found in {cfg.models_dir}. Run train.py first."
    )


def _resolve_champion_type(cfg: Settings) -> str:
    """Query MLflow for the champion model type tag; fall back to 'lstm_autoencoder'."""
    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        client = MlflowClient()
        version = client.get_model_version_by_alias(cfg.mlflow_model_name, cfg.mlflow_champion_alias)
        return client.get_model_version(cfg.mlflow_model_name, version.version).tags.get("model_type", "lstm_autoencoder")
    except Exception:
        log.debug("MLflow unreachable — defaulting champion type to lstm_autoencoder")
        return "lstm_autoencoder"


# ── prediction helpers ────────────────────────────────────────────────────────


def _predict_if(
    readings: np.ndarray,
    state: _ModelState,
    cfg: Settings,
) -> tuple[np.ndarray, np.ndarray]:
    """Score readings with the Isolation Forest.

    Returns:
        Tuple of ``(timestep_indices, scores)`` both shape ``(n_valid,)``.
    """
    scaled = state.scaler.transform(readings)
    flat = add_lag_features(scaled, cfg.n_lag_features)
    raw_scores = -state.model.decision_function(flat)
    # Dynamic threshold: 95th percentile of this batch's scores as a fallback;
    # production would use a precomputed training percentile stored in the artefact.
    threshold = float(np.percentile(raw_scores, 100 * (1 - cfg.if_contamination)))
    timesteps = np.arange(cfg.n_lag_features, len(readings), dtype=np.int32)
    return timesteps, raw_scores, threshold


def _predict_lstm(
    readings: np.ndarray,
    state: _ModelState,
    cfg: Settings,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Score readings with the LSTM Autoencoder.

    Returns:
        Tuple of ``(timestep_indices, scores, threshold)`` where timestep_indices
        are the *last* timestep of each window.
    """
    scaled = state.scaler.transform(readings)
    windows = sliding_window(scaled, cfg.window_size, cfg.step_size)
    if windows.shape[0] == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32), state.threshold
    scores = score_lstm(state.model, windows)
    # Each window's score is assigned to its last timestep.
    last_timesteps = np.arange(
        cfg.window_size - 1,
        cfg.window_size - 1 + len(windows) * cfg.step_size,
        cfg.step_size,
        dtype=np.int32,
    )
    return last_timesteps, scores, state.threshold


# ── request / response schemas ────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Payload for ``POST /predict``."""

    channel_id: str = Field(..., description="Telemetry channel identifier, e.g. 'P-1'.")
    readings: list[list[float]] = Field(
        ...,
        description=(
            "2-D array of sensor readings. Shape: (n_timesteps, n_features). "
            "Each inner list is one timestep of raw (unscaled) sensor values."
        ),
    )

    @field_validator("readings")
    @classmethod
    def _validate_readings(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) == 0:
            raise ValueError("readings must not be empty.")
        n = len(v[0])
        if any(len(row) != n for row in v):
            raise ValueError("All timesteps must have the same number of features.")
        return v


class PredictionPoint(BaseModel):
    """Anomaly score for a single timestep."""

    timestep: int
    score: float
    is_anomaly: bool


class PredictResponse(BaseModel):
    """Response from ``POST /predict``."""

    channel_id: str
    predictions: list[PredictionPoint]
    model_type: str
    model_alias: str
    n_readings_received: int
    n_predictions_returned: int


class HealthResponse(BaseModel):
    """Response from ``GET /health``."""

    status: str
    model_type: str | None
    model_alias: str | None
    api_version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Response from ``GET /metrics``."""

    total_requests: int
    total_anomalies: int
    anomaly_rate: float
    mean_score: float
    uptime_seconds: float


# ── app ───────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once at startup."""
    global _state
    cfg = get_settings()
    try:
        _state = _load_model_state(cfg)
        log.info("model loaded: %s", _state.model_type)
    except RuntimeError as exc:
        log.warning("startup: %s — /predict will return 503 until a model is available", exc)
    yield


def create_app() -> FastAPI:
    """Construct and return the FastAPI application."""
    cfg = get_settings()
    return FastAPI(
        title=cfg.api_title,
        version=cfg.api_version,
        lifespan=lifespan,
    )


app = create_app()


# ── endpoints ─────────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    """Score raw sensor readings and return per-timestep anomaly predictions.

    The service applies the same scaling and feature construction used during
    training, so callers send raw (unscaled) readings.

    Raises:
        503: No model is currently loaded.
        422: Insufficient timesteps for the active model's context window.
    """
    global _state, _metrics

    if _state is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run train.py first.")

    cfg = get_settings()
    readings = np.array(payload.readings, dtype=np.float32)
    n_timesteps = len(readings)

    # Validate minimum length for the active model.
    min_len = cfg.window_size if _state.model_type == "lstm_autoencoder" else cfg.n_lag_features + 1
    if n_timesteps < min_len:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{_state.model_type} requires at least {min_len} timesteps; "
                f"received {n_timesteps}."
            ),
        )

    if _state.model_type == "lstm_autoencoder":
        timesteps, scores, threshold = _predict_lstm(readings, _state, cfg)
    else:
        timesteps, scores, threshold = _predict_if(readings, _state, cfg)

    predictions = [
        PredictionPoint(
            timestep=int(t),
            score=float(s),
            is_anomaly=bool(s >= threshold),
        )
        for t, s in zip(timesteps, scores)
    ]

    n_anomalies = sum(p.is_anomaly for p in predictions)
    _metrics.total_requests += 1
    _metrics.total_anomalies += n_anomalies
    _metrics.score_sum += float(scores.mean()) if len(scores) else 0.0

    return PredictResponse(
        channel_id=payload.channel_id,
        predictions=predictions,
        model_type=_state.model_type,
        model_alias=_state.alias,
        n_readings_received=n_timesteps,
        n_predictions_returned=len(predictions),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service liveness and loaded model metadata."""
    cfg = get_settings()
    return HealthResponse(
        status="ok" if _state is not None else "degraded",
        model_type=_state.model_type if _state else None,
        model_alias=_state.alias if _state else None,
        api_version=cfg.api_version,
        uptime_seconds=time.time() - _metrics.started_at,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Return aggregate prediction statistics since service startup."""
    total = _metrics.total_requests
    return MetricsResponse(
        total_requests=total,
        total_anomalies=_metrics.total_anomalies,
        anomaly_rate=_metrics.total_anomalies / total if total else 0.0,
        mean_score=_metrics.score_sum / total if total else 0.0,
        uptime_seconds=time.time() - _metrics.started_at,
    )
