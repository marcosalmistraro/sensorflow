"""Drift monitoring and retraining flag logic.

Compares the training distribution against current data (test set as a
production proxy) using a per-feature two-sample Kolmogorov-Smirnov test.
If the fraction of drifted features exceeds ``drift_threshold``, a retrain
flag is written to ``reports/retrain_flag.json``.

The GitHub Actions retrain workflow reads that flag file and conditionally
triggers the full retrain → evaluate → promote pipeline.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pandas as pd
from config import Settings, get_settings
from scipy import stats

log = logging.getLogger(__name__)


# ── types ─────────────────────────────────────────────────────────────────────


class FeatureDriftResult(TypedDict):
    statistic: float   # KS test statistic
    p_value: float     # two-sided p-value
    drifted: bool      # True when p_value < drift_p_value


class DriftReport(TypedDict):
    retrain_required: bool
    drifted_features: int
    total_features: int
    drift_fraction: float
    drift_fraction_threshold: float
    per_feature: dict[str, FeatureDriftResult]
    timestamp: str


# ── core detection ────────────────────────────────────────────────────────────


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feat_cols: list[str],
    p_value_threshold: float,
) -> dict[str, FeatureDriftResult]:
    """Run a two-sample KS test for each feature column.

    Args:
        reference: Training-split DataFrame (the reference distribution).
        current: Current-data DataFrame (production proxy).
        feat_cols: Feature column names to test — must exist in both DataFrames.
        p_value_threshold: Features with p-value below this are flagged drifted.

    Returns:
        Dict mapping feature name to its :class:`FeatureDriftResult`.
    """
    results: dict[str, FeatureDriftResult] = {}

    for col in feat_cols:
        ref_vals = reference[col].dropna().values
        cur_vals = current[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            log.warning("skipping %s — empty after dropna", col)
            results[col] = FeatureDriftResult(statistic=0.0, p_value=1.0, drifted=False)
            continue

        ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
        results[col] = FeatureDriftResult(
            statistic=float(ks_stat),
            p_value=float(p_val),
            drifted=bool(p_val < p_value_threshold),
        )

    return results


def build_drift_report(
    per_feature: dict[str, FeatureDriftResult],
    drift_fraction_threshold: float,
) -> DriftReport:
    """Aggregate per-feature results into a summary drift report.

    Args:
        per_feature: Output of :func:`detect_drift`.
        drift_fraction_threshold: Fraction of drifted features that triggers
            ``retrain_required = True``.

    Returns:
        Populated :class:`DriftReport`.
    """
    total = len(per_feature)
    n_drifted = sum(1 for r in per_feature.values() if r["drifted"])
    drift_fraction = n_drifted / total if total else 0.0
    retrain_required = drift_fraction >= drift_fraction_threshold

    return DriftReport(
        retrain_required=retrain_required,
        drifted_features=n_drifted,
        total_features=total,
        drift_fraction=round(drift_fraction, 4),
        drift_fraction_threshold=drift_fraction_threshold,
        per_feature=per_feature,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── flag file I/O ─────────────────────────────────────────────────────────────


def write_flag(report: DriftReport, cfg: Settings) -> Path:
    """Serialise the drift report to ``reports/retrain_flag.json``.

    Args:
        report: Completed drift report.
        cfg: Settings instance.

    Returns:
        Path to the written flag file.
    """
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    flag_path = cfg.reports_dir / "retrain_flag.json"
    flag_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(
        "flag written → %s  (retrain_required=%s, drift=%.1f%%)",
        flag_path,
        report["retrain_required"],
        report["drift_fraction"] * 100,
    )
    return flag_path


def read_flag(cfg: Settings) -> DriftReport | None:
    """Load the most recent drift report from disk.

    Returns:
        Parsed :class:`DriftReport`, or ``None`` if the flag file does not exist.
    """
    flag_path = cfg.reports_dir / "retrain_flag.json"
    if not flag_path.exists():
        return None
    return json.loads(flag_path.read_text(encoding="utf-8"))


# ── orchestrator ──────────────────────────────────────────────────────────────


def run_monitor(cfg: Settings | None = None) -> DriftReport:
    """Compare train vs test distributions and write a retrain flag.

    Args:
        cfg: Settings instance; uses the module singleton when ``None``.

    Returns:
        The completed :class:`DriftReport`.
    """
    if cfg is None:
        cfg = get_settings()

    train_path = cfg.data_dir / "train.parquet"
    test_path = cfg.data_dir / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Parquet splits not found in {cfg.data_dir}. Run ingestion.py first."
        )

    log.info("loading reference (train) and current (test) distributions …")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    feat_cols = sorted(c for c in train_df.columns if c.startswith("f_"))
    if not feat_cols:
        raise ValueError("No feature columns found in train parquet.")

    log.info("running KS tests on %d features …", len(feat_cols))
    per_feature = detect_drift(train_df, test_df, feat_cols, cfg.drift_p_value)

    report = build_drift_report(per_feature, cfg.drift_threshold)

    n = report["drifted_features"]
    total = report["total_features"]
    log.info(
        "%d / %d features drifted (%.1f%%) — retrain_required=%s",
        n, total, report["drift_fraction"] * 100, report["retrain_required"],
    )

    write_flag(report, cfg)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_monitor()
    log.info("done: retrain_required=%s", result["retrain_required"])
