"""Unit tests for src/train.py — Isolation Forest and LSTM Autoencoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.ensemble import IsolationForest

from train import (
    LSTMAutoencoder,
    separation_score,
    train_isolation_forest,
    train_lstm_autoencoder,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_flat(n: int = 300, n_features: int = 25, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_features)).astype(np.float32)


def _make_labels(n: int, anomaly_frac: float = 0.05, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=bool)
    n_anom = max(1, int(n * anomaly_frac))
    idx = rng.choice(n, size=n_anom, replace=False)
    labels[idx] = True
    return labels


# ── separation_score ──────────────────────────────────────────────────────────


def test_separation_score_no_labels_returns_float() -> None:
    scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    result = separation_score(scores, labels=None)
    assert isinstance(result, float)


def test_separation_score_perfect_separation() -> None:
    # Anomalies have score 10, normals have score 0 → large positive separation
    scores = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
    labels = np.array([False, False, False, True, True])
    score = separation_score(scores, labels)
    assert score > 5.0


def test_separation_score_no_separation() -> None:
    # All scores identical → separation ≈ 0
    scores = np.ones(100)
    labels = np.zeros(100, dtype=bool)
    labels[:10] = True
    score = separation_score(scores, labels)
    assert abs(score) < 0.01


def test_separation_score_no_anomalies_in_labels() -> None:
    # All-normal labels → falls back to mean-based score, should return float
    scores = np.array([0.2, 0.3, 0.4])
    labels = np.zeros(3, dtype=bool)
    result = separation_score(scores, labels)
    assert isinstance(result, float)


def test_separation_score_higher_is_better() -> None:
    # Model A: scores 5 for anomalies, 0 for normals
    # Model B: scores 1 for anomalies, 0 for normals
    labels = np.array([False, False, True])
    score_a = separation_score(np.array([0.0, 0.0, 5.0]), labels)
    score_b = separation_score(np.array([0.0, 0.0, 1.0]), labels)
    assert score_a > score_b


# ── train_isolation_forest ────────────────────────────────────────────────────


def test_train_isolation_forest_returns_model(tmp_settings) -> None:
    train_flat = _make_flat(300)
    val_flat = _make_flat(100, seed=1)
    model, score = train_isolation_forest(train_flat, val_flat, None, tmp_settings)
    assert isinstance(model, IsolationForest)
    assert isinstance(score, float)


def test_train_isolation_forest_model_is_fitted(tmp_settings) -> None:
    train_flat = _make_flat(300)
    val_flat = _make_flat(100, seed=1)
    model, _ = train_isolation_forest(train_flat, val_flat, None, tmp_settings)
    # A fitted model can call predict without raising
    preds = model.predict(val_flat)
    assert set(preds).issubset({-1, 1})


def test_train_isolation_forest_with_labels_returns_score(tmp_settings) -> None:
    train_flat = _make_flat(300)
    val_flat = _make_flat(100, seed=1)
    val_labels = _make_labels(100)
    model, score = train_isolation_forest(train_flat, val_flat, val_labels, tmp_settings)
    assert isinstance(score, float)


def test_train_isolation_forest_respects_n_estimators(tmp_settings) -> None:
    train_flat = _make_flat(200)
    val_flat = _make_flat(50, seed=2)
    model, _ = train_isolation_forest(train_flat, val_flat, None, tmp_settings)
    assert model.n_estimators == tmp_settings.if_n_estimators


def test_train_isolation_forest_reproducible(tmp_settings) -> None:
    train_flat = _make_flat(200)
    val_flat = _make_flat(50, seed=2)
    _, score_a = train_isolation_forest(train_flat, val_flat, None, tmp_settings)
    _, score_b = train_isolation_forest(train_flat, val_flat, None, tmp_settings)
    assert score_a == score_b


def test_train_isolation_forest_anomaly_scores_higher_for_outliers(tmp_settings) -> None:
    """Injected extreme outliers should receive higher anomaly scores."""
    rng = np.random.default_rng(42)
    normal = rng.standard_normal((500, 25)).astype(np.float32)
    outliers = (rng.standard_normal((20, 25)) * 10 + 50).astype(np.float32)

    model, _ = train_isolation_forest(normal, normal, None, tmp_settings)

    normal_scores = -model.decision_function(normal)
    outlier_scores = -model.decision_function(outliers)

    assert outlier_scores.mean() > normal_scores.mean()


# ── LSTMAutoencoder (architecture) ────────────────────────────────────────────


def _make_windows(n: int = 64, seq_len: int = 16, n_features: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, seq_len, n_features)).astype(np.float32)


def test_lstm_autoencoder_output_shape() -> None:
    model = LSTMAutoencoder(n_features=4, hidden_size=8, num_layers=1, seq_len=16)
    x = torch.from_numpy(_make_windows(8)).float()
    out = model(x)
    assert out.shape == x.shape


def test_lstm_autoencoder_reconstruction_error_shape() -> None:
    model = LSTMAutoencoder(n_features=4, hidden_size=8, num_layers=1, seq_len=16)
    x = torch.from_numpy(_make_windows(8)).float()
    errors = model.reconstruction_error(x)
    assert errors.shape == (8,)


def test_lstm_autoencoder_reconstruction_error_non_negative() -> None:
    model = LSTMAutoencoder(n_features=4, hidden_size=8, num_layers=1, seq_len=16)
    x = torch.from_numpy(_make_windows(8)).float()
    errors = model.reconstruction_error(x)
    assert (errors >= 0).all()


def test_lstm_autoencoder_trained_error_less_than_untrained() -> None:
    """Reconstruction error on training data should decrease after training."""
    rng = np.random.default_rng(0)
    # Simple sine pattern — easy to learn
    t = np.linspace(0, 4 * np.pi, 500)
    signal = np.sin(t).astype(np.float32).reshape(-1, 1)
    windows = np.lib.stride_tricks.sliding_window_view(signal[:, 0], 16)[::1].copy()
    windows = windows[:, :, np.newaxis]  # (n, 16, 1)

    model_before = LSTMAutoencoder(n_features=1, hidden_size=16, num_layers=1, seq_len=16)
    x = torch.from_numpy(windows).float()
    error_before = model_before.reconstruction_error(x).mean().item()

    # Train for a few steps
    opt = torch.optim.Adam(model_before.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        loss = ((model_before(x) - x) ** 2).mean()
        loss.backward()
        opt.step()

    error_after = model_before.reconstruction_error(x).mean().item()
    assert error_after < error_before


# ── train_lstm_autoencoder ────────────────────────────────────────────────────


@pytest.fixture()
def tiny_lstm_settings(tmp_settings):
    """Settings with minimal LSTM config for fast tests."""
    tmp_settings.lstm_hidden_size = 8
    tmp_settings.lstm_num_layers = 1
    tmp_settings.lstm_epochs = 3
    tmp_settings.lstm_batch_size = 32
    tmp_settings.lstm_patience = 10
    return tmp_settings


def test_train_lstm_returns_model_and_threshold(tiny_lstm_settings) -> None:
    train_w = _make_windows(100, seq_len=16, n_features=4)
    val_w = _make_windows(30, seq_len=16, n_features=4)
    model, threshold, score = train_lstm_autoencoder(train_w, val_w, None, tiny_lstm_settings)
    assert isinstance(model, LSTMAutoencoder)
    assert threshold > 0.0
    assert isinstance(score, float)


def test_train_lstm_threshold_is_percentile_of_train_errors(tiny_lstm_settings) -> None:
    train_w = _make_windows(100, seq_len=16, n_features=4)
    val_w = _make_windows(30, seq_len=16, n_features=4)
    model, threshold, _ = train_lstm_autoencoder(train_w, val_w, None, tiny_lstm_settings)

    x = torch.from_numpy(train_w).float()
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(x).numpy()

    expected = float(
        np.percentile(errors, tiny_lstm_settings.lstm_reconstruction_threshold_percentile)
    )
    assert abs(threshold - expected) < 1e-5


def test_train_lstm_outliers_exceed_threshold(tiny_lstm_settings) -> None:
    """Windows of pure noise should score above a model trained on smooth data."""
    rng = np.random.default_rng(1)
    t = np.linspace(0, 8 * np.pi, 600)
    smooth = np.stack([np.sin(t), np.cos(t), np.sin(2 * t), np.cos(2 * t)], axis=1)
    # Build non-overlapping windows
    windows = smooth[: 576].reshape(36, 16, 4).astype(np.float32)

    # Bump epochs so the model actually learns the pattern
    tiny_lstm_settings.lstm_epochs = 15
    model, threshold, _ = train_lstm_autoencoder(windows, windows[:10], None, tiny_lstm_settings)

    noise = rng.standard_normal((10, 16, 4)).astype(np.float32) * 5
    x_noise = torch.from_numpy(noise).float()
    model.eval()
    with torch.no_grad():
        noise_errors = model.reconstruction_error(x_noise).numpy()

    assert noise_errors.mean() > threshold
