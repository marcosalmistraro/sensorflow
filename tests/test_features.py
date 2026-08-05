"""Unit tests for src/features.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from features import (
    add_lag_features,
    build_flat,
    build_windows,
    fit_scaler,
    scale_df,
    sliding_window,
    window_labels,
)
from ingestion import feature_columns


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_scaled_df(
    n_timesteps: int = 150,
    n_features: int = 25,
    chan_id: str = "P-1",
    split: str = "train",
    with_labels: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((n_timesteps, n_features)).astype(np.float32)
    feat_cols = feature_columns(n_features)
    df = pd.DataFrame(arr, columns=feat_cols)
    df.insert(0, "timestep", np.arange(n_timesteps, dtype=np.int32))
    df.insert(0, "channel_id", chan_id)
    df["split"] = split
    if with_labels:
        labels = np.zeros(n_timesteps, dtype=bool)
        labels[10:20] = True
        df["is_anomaly"] = labels
    return df


# ── sliding_window ────────────────────────────────────────────────────────────


def test_sliding_window_shape() -> None:
    arr = np.ones((100, 25), dtype=np.float32)
    out = sliding_window(arr, window_size=64, step_size=1)
    assert out.shape == (37, 64, 25)


def test_sliding_window_step_size() -> None:
    arr = np.ones((100, 25), dtype=np.float32)
    out = sliding_window(arr, window_size=10, step_size=5)
    expected_n = (100 - 10) // 5 + 1
    assert out.shape[0] == expected_n


def test_sliding_window_values_correct() -> None:
    arr = np.arange(20, dtype=np.float32).reshape(20, 1)
    out = sliding_window(arr, window_size=3, step_size=1)
    # First window should be [0, 1, 2]
    np.testing.assert_array_equal(out[0, :, 0], [0, 1, 2])
    # Second window should be [1, 2, 3]
    np.testing.assert_array_equal(out[1, :, 0], [1, 2, 3])


def test_sliding_window_too_short_returns_empty() -> None:
    arr = np.ones((5, 25), dtype=np.float32)
    out = sliding_window(arr, window_size=64, step_size=1)
    assert out.shape == (0, 64, 25)


def test_sliding_window_exact_fit() -> None:
    arr = np.ones((64, 25), dtype=np.float32)
    out = sliding_window(arr, window_size=64, step_size=1)
    assert out.shape[0] == 1


# ── window_labels ─────────────────────────────────────────────────────────────


def test_window_labels_anomaly_in_window() -> None:
    labels = np.zeros(100, dtype=bool)
    labels[10] = True
    win_lbls = window_labels(labels, window_size=64, step_size=1)
    # Window starting at 0 covers timesteps 0-63 → includes index 10 → anomalous
    assert win_lbls[0]


def test_window_labels_no_anomaly() -> None:
    labels = np.zeros(100, dtype=bool)
    win_lbls = window_labels(labels, window_size=64, step_size=1)
    assert not win_lbls.any()


def test_window_labels_length_matches_windows() -> None:
    arr = np.ones((100, 5), dtype=np.float32)
    labels = np.zeros(100, dtype=bool)
    windows = sliding_window(arr, window_size=10, step_size=3)
    win_lbls = window_labels(labels, window_size=10, step_size=3)
    assert len(win_lbls) == windows.shape[0]


def test_window_labels_conservative_any_rule() -> None:
    """A window with even one anomalous timestep must be labelled True."""
    labels = np.zeros(20, dtype=bool)
    labels[9] = True  # last timestep of first window [0:10]
    win_lbls = window_labels(labels, window_size=10, step_size=10)
    assert win_lbls[0]
    assert not win_lbls[1]


# ── add_lag_features ──────────────────────────────────────────────────────────


def test_add_lag_features_shape() -> None:
    arr = np.ones((100, 25), dtype=np.float32)
    out = add_lag_features(arr, n_lags=5)
    assert out.shape == (95, 25 * 6)


def test_add_lag_features_zero_lags_identity() -> None:
    arr = np.ones((100, 25), dtype=np.float32)
    out = add_lag_features(arr, n_lags=0)
    np.testing.assert_array_equal(out, arr)


def test_add_lag_features_values_correct() -> None:
    arr = np.arange(10, dtype=np.float32).reshape(10, 1)
    out = add_lag_features(arr, n_lags=2)
    # Row 0 of output = timestep 2: current=2, lag1=1, lag2=0
    assert out[0, 0] == 2.0
    assert out[0, 1] == 1.0
    assert out[0, 2] == 0.0


def test_add_lag_features_drops_first_n_rows() -> None:
    arr = np.ones((50, 3), dtype=np.float32)
    out = add_lag_features(arr, n_lags=3)
    assert len(out) == 47


# ── fit_scaler / scale_df ─────────────────────────────────────────────────────


def test_fit_scaler_returns_fitted_scaler() -> None:
    df = _make_scaled_df()
    feat_cols = feature_columns(25)
    scaler = fit_scaler(df, feat_cols)
    assert isinstance(scaler, StandardScaler)
    assert hasattr(scaler, "mean_")


def test_scale_df_zero_mean() -> None:
    df = _make_scaled_df(n_timesteps=1000)
    feat_cols = feature_columns(25)
    scaler = fit_scaler(df, feat_cols)
    scaled = scale_df(df, feat_cols, scaler)
    col_means = scaled[feat_cols].mean()
    np.testing.assert_allclose(col_means.values, 0.0, atol=1e-5)


def test_scale_df_does_not_modify_original() -> None:
    df = _make_scaled_df()
    feat_cols = feature_columns(25)
    scaler = fit_scaler(df, feat_cols)
    original_f00 = df["f_00"].values.copy()
    _ = scale_df(df, feat_cols, scaler)
    np.testing.assert_array_equal(df["f_00"].values, original_f00)


def test_scale_df_preserves_non_feature_columns() -> None:
    df = _make_scaled_df()
    feat_cols = feature_columns(25)
    scaler = fit_scaler(df, feat_cols)
    scaled = scale_df(df, feat_cols, scaler)
    pd.testing.assert_series_equal(df["channel_id"], scaled["channel_id"])
    pd.testing.assert_series_equal(df["timestep"], scaled["timestep"])


# ── build_windows ─────────────────────────────────────────────────────────────


def test_build_windows_shape() -> None:
    df = _make_scaled_df(n_timesteps=150)
    feat_cols = feature_columns(25)
    windows, labels = build_windows(df, feat_cols, window_size=64, step_size=1)
    assert windows.shape[1] == 64
    assert windows.shape[2] == 25
    assert labels is None


def test_build_windows_no_cross_channel_windows() -> None:
    """Windows from two channels of length 70 should never mix rows."""
    parts = [
        _make_scaled_df(n_timesteps=70, chan_id="P-1"),
        _make_scaled_df(n_timesteps=70, chan_id="S-1"),
    ]
    df = pd.concat(parts, ignore_index=True)
    feat_cols = feature_columns(25)
    # With window_size=64 and step_size=1 each channel contributes 7 windows
    windows, _ = build_windows(df, feat_cols, window_size=64, step_size=1)
    assert windows.shape[0] == 14


def test_build_windows_returns_labels_for_test() -> None:
    df = _make_scaled_df(n_timesteps=150, with_labels=True, split="test")
    feat_cols = feature_columns(25)
    windows, labels = build_windows(df, feat_cols, 64, 1, label_col="is_anomaly")
    assert labels is not None
    assert len(labels) == windows.shape[0]


# ── build_flat ────────────────────────────────────────────────────────────────


def test_build_flat_shape() -> None:
    df = _make_scaled_df(n_timesteps=100)
    feat_cols = feature_columns(25)
    flat, labels = build_flat(df, feat_cols, n_lags=5)
    assert flat.shape == (95, 25 * 6)
    assert labels is None


def test_build_flat_no_cross_channel_lags() -> None:
    """The first lag row of channel S-1 must not contain values from P-1."""
    parts = [
        _make_scaled_df(n_timesteps=50, chan_id="P-1"),
        _make_scaled_df(n_timesteps=50, chan_id="S-1"),
    ]
    df = pd.concat(parts, ignore_index=True)
    feat_cols = feature_columns(25)
    # With n_lags=2, each channel contributes 48 rows → 96 total
    flat, _ = build_flat(df, feat_cols, n_lags=2)
    assert flat.shape[0] == 96


def test_build_flat_returns_labels_for_test() -> None:
    df = _make_scaled_df(n_timesteps=100, with_labels=True, split="test")
    feat_cols = feature_columns(25)
    flat, labels = build_flat(df, feat_cols, n_lags=3, label_col="is_anomaly")
    assert labels is not None
    assert len(labels) == flat.shape[0]
