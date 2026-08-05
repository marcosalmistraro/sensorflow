"""Unit tests for src/train.py — Isolation Forest training."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import IsolationForest

from train import separation_score, train_isolation_forest


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
