"""Unit tests for src/evaluate.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from evaluate import (
    compute_metrics,
    load_lstm_autoencoder,
    score_isolation_forest,
    score_lstm,
)
from train import LSTMAutoencoder

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_scores(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n).astype(np.float32)


def _make_labels(n: int = 200, anomaly_frac: float = 0.1, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=bool)
    n_anom = max(1, int(n * anomaly_frac))
    labels[rng.choice(n, n_anom, replace=False)] = True
    return labels


# ── compute_metrics ───────────────────────────────────────────────────────────


def test_compute_metrics_perfect_classifier() -> None:
    labels = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    threshold = 0.5
    m = compute_metrics(scores, labels, threshold)
    assert m["f1"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(1.0)


def test_compute_metrics_worst_classifier() -> None:
    labels = np.array([False, False, True, True])
    # Predicts normal for anomalies, anomalous for normals
    scores = np.array([0.9, 0.95, 0.1, 0.2])
    threshold = 0.5
    m = compute_metrics(scores, labels, threshold)
    assert m["f1"] == pytest.approx(0.0)
    assert m["recall"] == pytest.approx(0.0)


def test_compute_metrics_returns_all_keys() -> None:
    scores = _make_scores()
    labels = _make_labels()
    threshold = 0.5
    m = compute_metrics(scores, labels, threshold)
    assert set(m.keys()) == {"f1", "precision", "recall", "auroc", "auprc"}


def test_compute_metrics_values_in_range() -> None:
    scores = _make_scores()
    labels = _make_labels()
    m = compute_metrics(scores, labels, threshold=0.5)
    for key, val in m.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0,1]"


def test_compute_metrics_auroc_better_than_random() -> None:
    # Build a near-perfect scorer
    labels = _make_labels(200, anomaly_frac=0.1)
    scores = labels.astype(np.float32) + np.random.default_rng(0).random(200) * 0.01
    m = compute_metrics(scores, labels, threshold=0.5)
    assert m["auroc"] > 0.9


def test_compute_metrics_threshold_controls_precision_recall_tradeoff() -> None:
    labels = np.array([False, False, False, True, True])
    scores = np.array([0.2, 0.4, 0.6, 0.8, 0.95])
    # High threshold → fewer positives → higher precision, lower recall
    m_high = compute_metrics(scores, labels, threshold=0.75)
    m_low = compute_metrics(scores, labels, threshold=0.35)
    assert m_high["precision"] >= m_low["precision"]
    assert m_high["recall"] <= m_low["recall"]


# ── score_lstm ────────────────────────────────────────────────────────────────


def test_score_lstm_shape() -> None:
    model = LSTMAutoencoder(n_features=4, hidden_size=8, num_layers=1, seq_len=16)
    model.eval()
    windows = np.random.default_rng(0).standard_normal((50, 16, 4)).astype(np.float32)
    scores = score_lstm(model, windows, batch_size=32)
    assert scores.shape == (50,)


def test_score_lstm_non_negative() -> None:
    model = LSTMAutoencoder(n_features=4, hidden_size=8, num_layers=1, seq_len=16)
    model.eval()
    windows = np.random.default_rng(0).standard_normal((20, 16, 4)).astype(np.float32)
    scores = score_lstm(model, windows)
    assert (scores >= 0).all()


def test_score_lstm_batching_consistent() -> None:
    """Scores should be identical regardless of batch size."""
    model = LSTMAutoencoder(n_features=4, hidden_size=8, num_layers=1, seq_len=16)
    model.eval()
    windows = np.random.default_rng(0).standard_normal((40, 16, 4)).astype(np.float32)
    scores_1 = score_lstm(model, windows, batch_size=10)
    scores_2 = score_lstm(model, windows, batch_size=40)
    np.testing.assert_allclose(scores_1, scores_2, rtol=1e-5)


# ── load_lstm_autoencoder ─────────────────────────────────────────────────────


def test_load_lstm_autoencoder_round_trip(tmp_settings) -> None:
    n_features, seq_len = 4, 16
    model = LSTMAutoencoder(n_features, tmp_settings.lstm_hidden_size, tmp_settings.lstm_num_layers, seq_len)
    threshold = 0.042

    tmp_settings.models_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_settings.models_dir / "lstm_autoencoder.pt"
    torch.save({"state_dict": model.state_dict(), "threshold": threshold}, path)

    loaded_model, loaded_threshold = load_lstm_autoencoder(tmp_settings, n_features, seq_len)

    assert abs(loaded_threshold - threshold) < 1e-6
    # Check one parameter is identical
    orig_param = next(iter(model.parameters())).detach().numpy()
    loaded_param = next(iter(loaded_model.parameters())).detach().numpy()
    np.testing.assert_allclose(orig_param, loaded_param)


def test_load_lstm_autoencoder_missing_file_raises(tmp_settings) -> None:
    with pytest.raises(FileNotFoundError):
        load_lstm_autoencoder(tmp_settings, n_features=4, seq_len=16)
