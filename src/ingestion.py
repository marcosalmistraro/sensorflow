"""Download, validate, and split the NASA SMAP telemetry dataset.

Pipeline
--------
1. Fetch ``labeled_anomalies.csv`` to discover channel IDs and anomaly intervals.
2. For each channel download ``{base_url}/train/{chan_id}.npy`` and
   ``{base_url}/test/{chan_id}.npy`` — plain numpy arrays, no pickle.
3. Build long-format DataFrames: one row per timestep, typed float32 feature
   columns, binary ``is_anomaly`` label on the test split.
4. Validate schema (shape, dtypes, NaN-free).
5. Chronologically split train → train / val, then write three parquet files.
"""

from __future__ import annotations

import ast
import io
import logging
from pathlib import Path
from typing import Iterator

import httpx
import numpy as np
import pandas as pd

from config import Settings, get_settings

log = logging.getLogger(__name__)


# ── column naming ─────────────────────────────────────────────────────────────


def feature_columns(n_features: int) -> list[str]:
    """Return ``['f_00', 'f_01', …, 'f_NN']`` for *n_features* columns."""
    return [f"f_{i:02d}" for i in range(n_features)]


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def _fetch_bytes(url: str, client: httpx.Client) -> bytes:
    """GET *url* and return the response body; raise on any HTTP error."""
    response = client.get(url, follow_redirects=True, timeout=120)
    response.raise_for_status()
    return response.content


def _download_npy(url: str, dest: Path, client: httpx.Client) -> np.ndarray:
    """Download a ``.npy`` file, persist it at *dest*, and return the array.

    If *dest* already exists the download is skipped (content-addressed by path
    convention, not hash — sufficient for a fixed public dataset).
    """
    if dest.exists():
        log.debug("cache hit: %s", dest.name)
        return np.load(dest, allow_pickle=False)

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    raw = _fetch_bytes(url, client)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    np.save(dest, arr)
    return arr


# ── labeled anomalies ─────────────────────────────────────────────────────────


def fetch_labels(cfg: Settings, client: httpx.Client) -> pd.DataFrame:
    """Download and parse ``labeled_anomalies.csv``.

    Returns a DataFrame with columns:
    ``chan_id, spacecraft, anomaly_sequences, class, num_values``
    where ``anomaly_sequences`` is a Python ``list[list[int]]``.
    """
    dest = cfg.data_dir / "labeled_anomalies.csv"

    if dest.exists():
        text = dest.read_text(encoding="utf-8")
    else:
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        log.info("downloading labels from %s", cfg.smap_labels_url)
        text = _fetch_bytes(cfg.smap_labels_url, client).decode("utf-8")
        dest.write_text(text, encoding="utf-8")

    df = pd.read_csv(io.StringIO(text))
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    return df


def _anomaly_mask(n_timesteps: int, sequences: list[list[int]]) -> np.ndarray:
    """Return a boolean mask of length *n_timesteps* from ``[start, end]`` pairs.

    Intervals are inclusive on both ends, matching the telemanom convention.
    """
    mask = np.zeros(n_timesteps, dtype=bool)
    for start, end in sequences:
        mask[start : end + 1] = True
    return mask


# ── per-channel construction ──────────────────────────────────────────────────


def build_channel_df(
    chan_id: str,
    array: np.ndarray,
    split: str,
    anomaly_sequences: list[list[int]] | None = None,
) -> pd.DataFrame:
    """Convert a raw numpy array for one channel into a typed DataFrame.

    Args:
        chan_id: Channel identifier (e.g. ``"P-1"``).
        array: Shape ``(n_timesteps, n_features)``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        anomaly_sequences: Anomaly ``[start, end]`` pairs; required for test.

    Returns:
        DataFrame columns: ``channel_id, timestep, f_00…f_NN[, is_anomaly], split``.
    """
    n_timesteps, n_features = array.shape
    cols = feature_columns(n_features)

    df = pd.DataFrame(array.astype(np.float32), columns=cols)
    df.insert(0, "timestep", np.arange(n_timesteps, dtype=np.int32))
    df.insert(0, "channel_id", chan_id)
    df["split"] = split

    if anomaly_sequences is not None:
        df["is_anomaly"] = _anomaly_mask(n_timesteps, anomaly_sequences)

    return df


def _iter_channel_arrays(
    labels_df: pd.DataFrame,
    cfg: Settings,
    client: httpx.Client,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, list[list[int]]]]:
    """Yield ``(chan_id, train_arr, test_arr, anomaly_sequences)`` per channel.

    Filters to ``cfg.smap_spacecraft`` unless that value is empty, in which
    case all spacecraft are included.
    """
    if cfg.smap_spacecraft:
        rows = labels_df[labels_df["spacecraft"] == cfg.smap_spacecraft]
    else:
        rows = labels_df

    base = cfg.smap_base_url.rstrip("/")
    chan_dir = cfg.data_dir / "channels"

    for _, row in rows.iterrows():
        chan_id: str = row["chan_id"]

        train_arr = _download_npy(
            f"{base}/train/{chan_id}.npy",
            chan_dir / "train" / f"{chan_id}.npy",
            client,
        )
        test_arr = _download_npy(
            f"{base}/test/{chan_id}.npy",
            chan_dir / "test" / f"{chan_id}.npy",
            client,
        )

        yield chan_id, train_arr, test_arr, row["anomaly_sequences"]


# ── validation ────────────────────────────────────────────────────────────────


def validate_split(df: pd.DataFrame, split: str, n_features: int) -> None:
    """Raise ``ValueError`` if *df* does not match the expected schema.

    Checks:
    - Required structural columns are present.
    - Feature columns ``f_00…f_NN`` exist and are ``float32``.
    - Zero NaN values in feature columns.
    - ``is_anomaly`` column is present on the test split.
    """
    required = {"channel_id", "timestep", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{split}] missing columns: {missing}")

    feat_cols = feature_columns(n_features)
    missing_feats = set(feat_cols) - set(df.columns)
    if missing_feats:
        raise ValueError(f"[{split}] missing feature columns: {missing_feats}")

    wrong_dtype = [c for c in feat_cols if df[c].dtype != np.float32]
    if wrong_dtype:
        raise ValueError(
            f"[{split}] feature columns with wrong dtype: "
            + ", ".join(f"{c}={df[c].dtype}" for c in wrong_dtype)
        )

    nan_totals = df[feat_cols].isna().sum()
    bad_cols = nan_totals[nan_totals > 0]
    if not bad_cols.empty:
        raise ValueError(f"[{split}] NaNs in: {bad_cols.to_dict()}")

    if split == "test" and "is_anomaly" not in df.columns:
        raise ValueError("[test] is_anomaly column missing")

    log.info(
        "validated %-5s — %d rows, %d channels",
        split,
        len(df),
        df["channel_id"].nunique(),
    )


# ── train / val split ─────────────────────────────────────────────────────────


def split_train_val(
    train_df: pd.DataFrame,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *train_df* into train and val by taking the tail of each channel.

    Splitting is strictly chronological — the final ``val_ratio`` fraction of
    each channel's timesteps become the validation set. This preserves the
    temporal ordering that the sliding-window feature builder relies on.
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []

    for _, grp in train_df.groupby("channel_id", sort=False):
        n_val = max(1, int(len(grp) * val_ratio))
        train_parts.append(grp.iloc[:-n_val])
        val_parts.append(grp.iloc[-n_val:].copy())

    train_out = pd.concat(train_parts, ignore_index=True)
    val_out = pd.concat(val_parts, ignore_index=True)
    val_out["split"] = "val"
    return train_out, val_out


# ── orchestrator ──────────────────────────────────────────────────────────────


def run_ingestion(cfg: Settings | None = None) -> dict[str, Path]:
    """Download SMAP data, validate schema, and write train/val/test parquet.

    Args:
        cfg: Settings instance; uses the module singleton when ``None``.

    Returns:
        Mapping of split name (``"train"``, ``"val"``, ``"test"``,
        ``"labels"``) to the written parquet path.
    """
    if cfg is None:
        cfg = get_settings()

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    log.info("ingestion start — data_dir=%s", cfg.data_dir.resolve())

    with httpx.Client() as client:
        labels_df = fetch_labels(cfg, client)
        log.info("labels: %d channel sequences", len(labels_df))

        raw_train: list[pd.DataFrame] = []
        raw_test: list[pd.DataFrame] = []
        n_features: int | None = None

        for chan_id, train_arr, test_arr, anomaly_seqs in _iter_channel_arrays(
            labels_df, cfg, client
        ):
            if n_features is None:
                n_features = train_arr.shape[1]
            elif train_arr.shape[1] != n_features:
                raise ValueError(
                    f"Channel {chan_id!r} has {train_arr.shape[1]} features; "
                    f"expected {n_features}."
                )

            raw_train.append(build_channel_df(chan_id, train_arr, "train"))
            raw_test.append(
                build_channel_df(chan_id, test_arr, "test", anomaly_sequences=anomaly_seqs)
            )

    if n_features is None:
        raise RuntimeError("No channels downloaded — check labels URL and network.")

    train_full = pd.concat(raw_train, ignore_index=True)
    test_df = pd.concat(raw_test, ignore_index=True)

    train_df, val_df = split_train_val(train_full, cfg.test_split_ratio)

    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        validate_split(df, split, n_features)

    paths: dict[str, Path] = {}
    for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dest = cfg.data_dir / f"{split}.parquet"
        df.to_parquet(dest, index=False)
        log.info("wrote %s — %d rows → %s", split, len(df), dest)
        paths[split] = dest

    labels_path = cfg.data_dir / "labels.parquet"
    labels_df.to_parquet(labels_path, index=False)
    paths["labels"] = labels_path

    log.info("ingestion complete — %s", {k: str(v) for k, v in paths.items()})
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_ingestion()
