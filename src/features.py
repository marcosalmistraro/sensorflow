"""Feature engineering: scaling, sliding windows, and lag features.

Takes the parquet splits from ingestion and writes ready-to-train numpy arrays:

  Windows  (n_windows,  window_size, n_features)     → LSTM Autoencoder
  Flat     (n_samples,  n_features × (1 + n_lags))   → Isolation Forest
  Labels   (n_windows,) / (n_samples,) bool           → evaluation only

The StandardScaler is fit *exclusively* on train data, then applied identically
to val and test, and serialised to disk so the serving path uses the same transform.

Windowing and lag construction are performed per channel so no window ever
crosses a channel boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from config import Settings, get_settings
from ingestion import feature_columns
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ── pure array transforms ─────────────────────────────────────────────────────


def sliding_window(arr: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """Cut a 2-D time series into overlapping windows.

    Args:
        arr: Shape ``(n_timesteps, n_features)``.
        window_size: Number of consecutive timesteps per window.
        step_size: Stride between the start of successive windows.

    Returns:
        Shape ``(n_windows, window_size, n_features)``.
        Returns an empty array with shape ``(0, window_size, n_features)`` when
        the channel is shorter than one window.
    """
    n_timesteps, n_features = arr.shape
    if n_timesteps < window_size:
        log.warning(
            "channel shorter than window_size (%d < %d) — skipping",
            n_timesteps,
            window_size,
        )
        return np.empty((0, window_size, n_features), dtype=arr.dtype)

    starts = range(0, n_timesteps - window_size + 1, step_size)
    return np.stack([arr[s : s + window_size] for s in starts])


def window_labels(labels: np.ndarray, window_size: int, step_size: int) -> np.ndarray:
    """Align a per-timestep boolean label vector to the windowed representation.

    A window is labelled anomalous if *any* timestep inside it is anomalous.
    This is the conservative choice: it avoids missing anomalies that only
    occupy a fraction of the window.

    Args:
        labels: Shape ``(n_timesteps,)`` boolean array.
        window_size: Must match the value used in :func:`sliding_window`.
        step_size: Must match the value used in :func:`sliding_window`.

    Returns:
        Shape ``(n_windows,)`` boolean array.
    """
    n = len(labels)
    if n < window_size:
        return np.empty(0, dtype=bool)

    starts = range(0, n - window_size + 1, step_size)
    return np.array([labels[s : s + window_size].any() for s in starts])


def add_lag_features(arr: np.ndarray, n_lags: int) -> np.ndarray:
    """Append lagged copies of every feature column.

    For each feature at time *t*, appends its value at *t-1*, *t-2*, …, *t-n_lags*.
    The first ``n_lags`` rows are dropped because they have no complete lag history.

    Args:
        arr: Shape ``(n_timesteps, n_features)``.
        n_lags: Number of lag steps. Returns *arr* unchanged when 0.

    Returns:
        Shape ``(n_timesteps - n_lags, n_features × (1 + n_lags))``.
    """
    if n_lags == 0:
        return arr

    n = len(arr)
    current = arr[n_lags:]  # (n - n_lags, n_features)
    lag_arrays = [arr[n_lags - lag : n - lag] for lag in range(1, n_lags + 1)]
    return np.concatenate([current, *lag_arrays], axis=1)


# ── scaler ────────────────────────────────────────────────────────────────────


def fit_scaler(train_df: pd.DataFrame, feat_cols: list[str]) -> StandardScaler:
    """Fit a StandardScaler on the training feature matrix.

    Args:
        train_df: Full training DataFrame (all channels concatenated).
        feat_cols: Column names of the feature columns to scale.

    Returns:
        Fitted :class:`~sklearn.preprocessing.StandardScaler`.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols].values)
    log.info("scaler fitted on %d train rows, %d features", len(train_df), len(feat_cols))
    return scaler


def scale_df(df: pd.DataFrame, feat_cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    """Return a copy of *df* with feature columns replaced by scaled values."""
    out = df.copy()
    out[feat_cols] = scaler.transform(df[feat_cols].values).astype(np.float32)
    return out


# ── per-split window / flat construction ──────────────────────────────────────


def build_windows(
    df: pd.DataFrame,
    feat_cols: list[str],
    window_size: int,
    step_size: int,
    label_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build windowed arrays from a scaled split DataFrame.

    Windowing is applied per channel so no window spans a boundary.

    Args:
        df: Scaled split DataFrame with ``channel_id`` and feature columns.
        feat_cols: Feature column names.
        window_size: Timesteps per window.
        step_size: Stride between windows.
        label_col: Optional column of per-timestep boolean labels. When
            provided, a corresponding label array is returned.

    Returns:
        Tuple ``(windows, labels)`` where *windows* has shape
        ``(n_windows, window_size, n_features)`` and *labels* is
        ``(n_windows,)`` bool or ``None``.
    """
    win_parts: list[np.ndarray] = []
    lbl_parts: list[np.ndarray] = []

    for _, grp in df.groupby("channel_id", sort=False):
        arr = grp[feat_cols].values.astype(np.float32)
        wins = sliding_window(arr, window_size, step_size)
        if wins.shape[0] == 0:
            continue
        win_parts.append(wins)

        if label_col is not None:
            lbls = grp[label_col].values.astype(bool)
            lbl_parts.append(window_labels(lbls, window_size, step_size))

    if not win_parts:
        n_feat = len(feat_cols)
        empty_win = np.empty((0, window_size, n_feat), dtype=np.float32)
        return empty_win, (np.empty(0, dtype=bool) if label_col else None)

    windows = np.concatenate(win_parts, axis=0)
    labels = np.concatenate(lbl_parts, axis=0) if lbl_parts else None
    return windows, labels


def build_flat(
    df: pd.DataFrame,
    feat_cols: list[str],
    n_lags: int,
    label_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Build flat feature arrays (with lags) from a scaled split DataFrame.

    Lag construction is applied per channel; the first ``n_lags`` rows of
    each channel are dropped to avoid NaN lag values.

    Args:
        df: Scaled split DataFrame with ``channel_id`` and feature columns.
        feat_cols: Feature column names.
        n_lags: Number of lag steps to append.
        label_col: Optional per-timestep label column.

    Returns:
        Tuple ``(flat, labels)`` where *flat* has shape
        ``(n_samples, n_features × (1 + n_lags))`` and *labels* is
        ``(n_samples,)`` bool or ``None``.
    """
    flat_parts: list[np.ndarray] = []
    lbl_parts: list[np.ndarray] = []

    for _, grp in df.groupby("channel_id", sort=False):
        arr = grp[feat_cols].values.astype(np.float32)
        flat = add_lag_features(arr, n_lags)
        flat_parts.append(flat)

        if label_col is not None:
            lbls = grp[label_col].values.astype(bool)
            lbl_parts.append(lbls[n_lags:] if n_lags else lbls)

    if not flat_parts:
        n_cols = len(feat_cols) * (1 + n_lags)
        return np.empty((0, n_cols), dtype=np.float32), (np.empty(0, dtype=bool) if label_col else None)

    flat = np.concatenate(flat_parts, axis=0)
    labels = np.concatenate(lbl_parts, axis=0) if lbl_parts else None
    return flat, labels


# ── orchestrator ──────────────────────────────────────────────────────────────


def run_features(cfg: Settings | None = None) -> dict[str, Path]:
    """Scale, window, and lag-featurise all three parquet splits.

    Args:
        cfg: Settings instance; uses the module singleton when ``None``.

    Returns:
        Dict mapping artefact name to its written path.
    """
    if cfg is None:
        cfg = get_settings()

    features_dir = cfg.data_dir.parent / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    train_df = pd.read_parquet(cfg.data_dir / "train.parquet")
    val_df = pd.read_parquet(cfg.data_dir / "val.parquet")
    test_df = pd.read_parquet(cfg.data_dir / "test.parquet")

    n_raw_features = len([c for c in train_df.columns if c.startswith("f_")])
    feat_cols = feature_columns(n_raw_features)
    log.info("feature columns: %d", n_raw_features)

    # ── scale ─────────────────────────────────────────────────────────────────
    scaler = fit_scaler(train_df, feat_cols)
    train_df = scale_df(train_df, feat_cols, scaler)
    val_df = scale_df(val_df, feat_cols, scaler)
    test_df = scale_df(test_df, feat_cols, scaler)

    scaler_path = cfg.models_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info("scaler saved → %s", scaler_path)

    paths: dict[str, Path] = {"scaler": scaler_path}

    # ── windows (LSTM) ────────────────────────────────────────────────────────
    for split, df, lbl_col in [
        ("train", train_df, None),
        ("val", val_df, None),
        ("test", test_df, "is_anomaly"),
    ]:
        windows, labels = build_windows(
            df, feat_cols, cfg.window_size, cfg.step_size, label_col=lbl_col
        )
        win_path = features_dir / f"{split}_windows.npy"
        np.save(win_path, windows)
        paths[f"{split}_windows"] = win_path
        log.info("%s windows: %s → %s", split, windows.shape, win_path)

        if labels is not None:
            lbl_path = features_dir / f"{split}_window_labels.npy"
            np.save(lbl_path, labels)
            paths[f"{split}_window_labels"] = lbl_path
            log.info("%s window labels: %s anomalies out of %s", split, labels.sum(), len(labels))

    # ── flat + lags (Isolation Forest) ────────────────────────────────────────
    for split, df, lbl_col in [
        ("train", train_df, None),
        ("val", val_df, None),
        ("test", test_df, "is_anomaly"),
    ]:
        flat, labels = build_flat(
            df, feat_cols, cfg.n_lag_features, label_col=lbl_col
        )
        flat_path = features_dir / f"{split}_flat.npy"
        np.save(flat_path, flat)
        paths[f"{split}_flat"] = flat_path
        log.info("%s flat: %s → %s", split, flat.shape, flat_path)

        if labels is not None:
            lbl_path = features_dir / f"{split}_flat_labels.npy"
            np.save(lbl_path, labels)
            paths[f"{split}_flat_labels"] = lbl_path

    log.info("feature engineering complete")
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_features()
