"""Unit tests for src/monitor.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitor import (
    build_drift_report,
    detect_drift,
    read_flag,
    write_flag,
)
from ingestion import feature_columns


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_df(
    n: int = 500,
    n_features: int = 5,
    loc: float = 0.0,
    scale: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.normal(loc, scale, (n, n_features)).astype(np.float32)
    return pd.DataFrame(arr, columns=feature_columns(n_features))


# ── detect_drift ──────────────────────────────────────────────────────────────


def test_detect_drift_identical_distributions_no_drift() -> None:
    df = _make_df(1000, seed=0)
    feat_cols = list(df.columns)
    results = detect_drift(df, df, feat_cols, p_value_threshold=0.05)
    drifted = [k for k, v in results.items() if v["drifted"]]
    # Identical data → p-value = 1 → no drift
    assert len(drifted) == 0


def test_detect_drift_very_different_distributions_all_drift() -> None:
    reference = _make_df(1000, loc=0.0, scale=1.0, seed=0)
    current = _make_df(1000, loc=10.0, scale=1.0, seed=1)  # completely shifted
    feat_cols = list(reference.columns)
    results = detect_drift(reference, current, feat_cols, p_value_threshold=0.05)
    drifted = [k for k, v in results.items() if v["drifted"]]
    assert len(drifted) == len(feat_cols)


def test_detect_drift_returns_all_features() -> None:
    ref = _make_df(200, n_features=5)
    cur = _make_df(200, n_features=5, seed=1)
    feat_cols = list(ref.columns)
    results = detect_drift(ref, cur, feat_cols, p_value_threshold=0.05)
    assert set(results.keys()) == set(feat_cols)


def test_detect_drift_result_fields() -> None:
    ref = _make_df(200)
    cur = _make_df(200, seed=1)
    feat_cols = list(ref.columns)
    results = detect_drift(ref, cur, feat_cols, p_value_threshold=0.05)
    for v in results.values():
        assert "statistic" in v
        assert "p_value" in v
        assert "drifted" in v
        assert 0.0 <= v["statistic"] <= 1.0
        assert 0.0 <= v["p_value"] <= 1.0
        assert isinstance(v["drifted"], bool)


def test_detect_drift_p_value_threshold_controls_sensitivity() -> None:
    rng = np.random.default_rng(42)
    ref = pd.DataFrame(rng.normal(0, 1, (500, 1)).astype(np.float32), columns=["f_00"])
    cur = pd.DataFrame(rng.normal(0.5, 1, (500, 1)).astype(np.float32), columns=["f_00"])

    strict = detect_drift(ref, cur, ["f_00"], p_value_threshold=0.5)
    lenient = detect_drift(ref, cur, ["f_00"], p_value_threshold=0.001)

    # Stricter threshold (higher p_value required to pass) catches drift more easily
    assert strict["f_00"]["drifted"] or not lenient["f_00"]["drifted"]


def test_detect_drift_statistic_larger_for_more_drift() -> None:
    ref = _make_df(500, loc=0.0, n_features=1, seed=0)
    slight = _make_df(500, loc=0.1, n_features=1, seed=1)
    large = _make_df(500, loc=5.0, n_features=1, seed=2)

    res_slight = detect_drift(ref, slight, ["f_00"], p_value_threshold=0.05)
    res_large = detect_drift(ref, large, ["f_00"], p_value_threshold=0.05)

    assert res_large["f_00"]["statistic"] > res_slight["f_00"]["statistic"]


# ── build_drift_report ────────────────────────────────────────────────────────


def test_build_drift_report_retrain_required_above_threshold() -> None:
    per_feature = {
        "f_00": {"statistic": 0.5, "p_value": 0.01, "drifted": True},
        "f_01": {"statistic": 0.5, "p_value": 0.01, "drifted": True},
        "f_02": {"statistic": 0.1, "p_value": 0.9, "drifted": False},
    }
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    # 2/3 = 0.67 > 0.5 → retrain required
    assert report["retrain_required"] is True
    assert report["drifted_features"] == 2
    assert report["total_features"] == 3


def test_build_drift_report_no_retrain_below_threshold() -> None:
    per_feature = {
        "f_00": {"statistic": 0.5, "p_value": 0.01, "drifted": True},
        "f_01": {"statistic": 0.1, "p_value": 0.9, "drifted": False},
        "f_02": {"statistic": 0.1, "p_value": 0.9, "drifted": False},
        "f_03": {"statistic": 0.1, "p_value": 0.9, "drifted": False},
    }
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    # 1/4 = 0.25 < 0.5 → no retrain
    assert report["retrain_required"] is False


def test_build_drift_report_has_timestamp() -> None:
    per_feature = {"f_00": {"statistic": 0.1, "p_value": 0.5, "drifted": False}}
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    assert "timestamp" in report
    assert report["timestamp"]  # non-empty


def test_build_drift_report_drift_fraction_correct() -> None:
    per_feature = {f"f_{i:02d}": {"statistic": 0.5, "p_value": 0.01, "drifted": i < 3}
                   for i in range(10)}
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    assert report["drift_fraction"] == pytest.approx(0.3)
    assert report["drifted_features"] == 3


# ── write_flag / read_flag ────────────────────────────────────────────────────


def test_write_flag_creates_file(tmp_settings) -> None:
    per_feature = {"f_00": {"statistic": 0.1, "p_value": 0.5, "drifted": False}}
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    path = write_flag(report, tmp_settings)
    assert path.exists()


def test_write_flag_valid_json(tmp_settings) -> None:
    per_feature = {"f_00": {"statistic": 0.1, "p_value": 0.5, "drifted": False}}
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    path = write_flag(report, tmp_settings)
    loaded = json.loads(path.read_text())
    assert "retrain_required" in loaded
    assert "per_feature" in loaded


def test_read_flag_round_trip(tmp_settings) -> None:
    per_feature = {"f_00": {"statistic": 0.42, "p_value": 0.03, "drifted": True}}
    report = build_drift_report(per_feature, drift_fraction_threshold=0.5)
    write_flag(report, tmp_settings)
    loaded = read_flag(tmp_settings)
    assert loaded is not None
    assert loaded["retrain_required"] == report["retrain_required"]
    assert loaded["drifted_features"] == report["drifted_features"]


def test_read_flag_returns_none_when_missing(tmp_settings) -> None:
    assert read_flag(tmp_settings) is None
