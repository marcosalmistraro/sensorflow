"""Unit tests for src/ingestion.py.

All tests are offline — no HTTP calls are made. Download helpers are tested
by pre-seeding the cache directory so _download_npy hits the cache path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ingestion import (
    _anomaly_mask,
    _generate_synthetic_channel,
    build_channel_df,
    feature_columns,
    split_train_val,
    validate_split,
)

# ── feature_columns ───────────────────────────────────────────────────────────


def test_feature_columns_count() -> None:
    cols = feature_columns(25)
    assert len(cols) == 25


def test_feature_columns_format() -> None:
    cols = feature_columns(3)
    assert cols == ["f_00", "f_01", "f_02"]


def test_feature_columns_zero_padded_past_nine() -> None:
    cols = feature_columns(11)
    assert cols[9] == "f_09"
    assert cols[10] == "f_10"


# ── _anomaly_mask ─────────────────────────────────────────────────────────────


def test_anomaly_mask_basic() -> None:
    mask = _anomaly_mask(10, [[2, 4]])
    assert not mask[1]
    assert mask[2]
    assert mask[4]
    assert not mask[5]


def test_anomaly_mask_inclusive_ends() -> None:
    mask = _anomaly_mask(5, [[0, 4]])
    assert mask.all()


def test_anomaly_mask_multiple_sequences() -> None:
    mask = _anomaly_mask(20, [[1, 3], [15, 17]])
    assert mask[1:4].all()
    assert mask[15:18].all()
    assert not mask[4]
    assert not mask[14]


def test_anomaly_mask_no_sequences() -> None:
    mask = _anomaly_mask(10, [])
    assert not mask.any()


# ── build_channel_df ──────────────────────────────────────────────────────────


def test_build_channel_df_shape(sample_channel_array: np.ndarray) -> None:
    df = build_channel_df("P-1", sample_channel_array, "train")
    assert len(df) == 200
    assert "channel_id" in df.columns
    assert "timestep" in df.columns
    assert "f_00" in df.columns
    assert "f_24" in df.columns


def test_build_channel_df_dtypes(sample_channel_array: np.ndarray) -> None:
    df = build_channel_df("P-1", sample_channel_array, "train")
    feat_cols = feature_columns(25)
    for col in feat_cols:
        assert df[col].dtype == np.float32, f"{col} dtype mismatch"


def test_build_channel_df_timestep_monotonic(sample_channel_array: np.ndarray) -> None:
    df = build_channel_df("P-1", sample_channel_array, "train")
    assert (df["timestep"].diff().dropna() == 1).all()


def test_build_channel_df_anomaly_column(sample_channel_array: np.ndarray) -> None:
    df = build_channel_df("P-1", sample_channel_array, "test", anomaly_sequences=[[10, 20]])
    assert "is_anomaly" in df.columns
    assert df.loc[10, "is_anomaly"]
    assert df.loc[20, "is_anomaly"]
    assert not df.loc[9, "is_anomaly"]


def test_build_channel_df_no_anomaly_on_train(sample_channel_array: np.ndarray) -> None:
    df = build_channel_df("P-1", sample_channel_array, "train")
    assert "is_anomaly" not in df.columns


def test_build_channel_df_split_label(sample_channel_array: np.ndarray) -> None:
    df = build_channel_df("P-1", sample_channel_array, "train")
    assert (df["split"] == "train").all()


# ── validate_split ────────────────────────────────────────────────────────────


def _make_valid_df(split: str, n: int = 100, with_anomaly: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((n, 25)).astype(np.float32)
    return build_channel_df("P-1", arr, split, anomaly_sequences=[[0, 5]] if with_anomaly else None)


def test_validate_split_passes_on_valid_train() -> None:
    df = _make_valid_df("train")
    validate_split(df, "train", 25)  # should not raise


def test_validate_split_passes_on_valid_test() -> None:
    df = _make_valid_df("test", with_anomaly=True)
    validate_split(df, "test", 25)  # should not raise


def test_validate_split_missing_column() -> None:
    df = _make_valid_df("train").drop(columns=["channel_id"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_split(df, "train", 25)


def test_validate_split_wrong_dtype() -> None:
    df = _make_valid_df("train")
    df["f_00"] = df["f_00"].astype(np.float64)
    with pytest.raises(ValueError, match="wrong dtype"):
        validate_split(df, "train", 25)


def test_validate_split_nan_raises() -> None:
    df = _make_valid_df("train")
    df.loc[0, "f_01"] = float("nan")
    with pytest.raises(ValueError, match="NaNs"):
        validate_split(df, "train", 25)


def test_validate_split_test_missing_anomaly_column() -> None:
    df = _make_valid_df("test")  # no anomaly_sequences passed → no is_anomaly col
    with pytest.raises(ValueError, match="is_anomaly column missing"):
        validate_split(df, "test", 25)


# ── split_train_val ───────────────────────────────────────────────────────────


def _make_train_df(n_per_channel: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    parts = []
    for chan in ["P-1", "S-1"]:
        arr = rng.standard_normal((n_per_channel, 25)).astype(np.float32)
        parts.append(build_channel_df(chan, arr, "train"))
    return pd.concat(parts, ignore_index=True)


def test_split_train_val_row_count() -> None:
    df = _make_train_df(100)
    train, val = split_train_val(df, val_ratio=0.2)
    # 2 channels × 100 rows → 80 train + 20 val per channel
    assert len(train) + len(val) == len(df)


def test_split_train_val_ratio() -> None:
    df = _make_train_df(100)
    _, val = split_train_val(df, val_ratio=0.2)
    # each channel contributes 20 val rows → 40 total
    assert len(val) == 40


def test_split_train_val_chronological_order() -> None:
    """Val rows must come strictly after train rows for each channel."""
    df = _make_train_df(100)
    train, val = split_train_val(df, val_ratio=0.2)
    for chan in ["P-1", "S-1"]:
        max_train_ts = train[train["channel_id"] == chan]["timestep"].max()
        min_val_ts = val[val["channel_id"] == chan]["timestep"].min()
        assert min_val_ts > max_train_ts


def test_split_train_val_split_label() -> None:
    df = _make_train_df(50)
    train, val = split_train_val(df, val_ratio=0.2)
    assert (train["split"] == "train").all()
    assert (val["split"] == "val").all()


def test_split_train_val_small_channel() -> None:
    """Even a single-row channel should produce at least one val row."""
    df = _make_train_df(1)
    train, val = split_train_val(df, val_ratio=0.2)
    for chan in df["channel_id"].unique():
        assert len(val[val["channel_id"] == chan]) >= 1


# ── _generate_synthetic_channel ───────────────────────────────────────────────


def test_synthetic_channel_shapes() -> None:
    train, test = _generate_synthetic_channel("P-1", n_train=400, n_test=100, n_features=25, anomaly_sequences=[])
    assert train.shape == (400, 25)
    assert test.shape == (100, 25)


def test_synthetic_channel_dtype() -> None:
    train, test = _generate_synthetic_channel("P-1", n_train=200, n_test=50, n_features=25, anomaly_sequences=[])
    assert train.dtype == np.float32
    assert test.dtype == np.float32


def test_synthetic_channel_deterministic() -> None:
    a_train, a_test = _generate_synthetic_channel("P-1", 200, 50, 25, [[10, 20]])
    b_train, b_test = _generate_synthetic_channel("P-1", 200, 50, 25, [[10, 20]])
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_test, b_test)


def test_synthetic_channel_different_per_channel() -> None:
    _, test_p1 = _generate_synthetic_channel("P-1", 200, 50, 25, [])
    _, test_s1 = _generate_synthetic_channel("S-1", 200, 50, 25, [])
    assert not np.allclose(test_p1, test_s1)


def test_synthetic_channel_anomaly_injected() -> None:
    """Anomaly window should have higher mean absolute value than baseline."""
    _, test = _generate_synthetic_channel("P-1", n_train=400, n_test=200, n_features=25, anomaly_sequences=[[50, 80]])
    anomaly_mag = np.abs(test[50:81]).mean()
    normal_mag = np.abs(test[100:150]).mean()
    assert anomaly_mag > normal_mag


def test_synthetic_channel_no_nan() -> None:
    train, test = _generate_synthetic_channel("P-1", 200, 100, 25, [[5, 15], [60, 70]])
    assert not np.isnan(train).any()
    assert not np.isnan(test).any()


def test_synthetic_channel_anomaly_clipped_to_bounds() -> None:
    """Anomaly sequences that exceed n_test should not raise and should be clipped."""
    train, test = _generate_synthetic_channel("P-1", 200, 50, 25, [[40, 200]])
    assert test.shape == (50, 25)
