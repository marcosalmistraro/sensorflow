"""Shared pytest fixtures for SensorFlow tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from config import Settings


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    """Settings instance pointing all paths at a temporary directory."""
    return Settings(
        data_dir=tmp_path / "data" / "raw",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        mlflow_tracking_uri="http://localhost:5000",
    )


@pytest.fixture()
def sample_channel_array() -> np.ndarray:
    """Small (200, 25) float32 array mimicking one SMAP channel."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 25)).astype(np.float32)


@pytest.fixture()
def sample_labels_df() -> pd.DataFrame:
    """Minimal labeled_anomalies DataFrame for two SMAP channels."""
    return pd.DataFrame(
        {
            "chan_id": ["P-1", "S-1"],
            "spacecraft": ["SMAP", "SMAP"],
            "anomaly_sequences": [[[10, 20]], [[50, 60]]],
            "class": ["point", "contextual"],
            "num_values": [200, 200],
        }
    )
