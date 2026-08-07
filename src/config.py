"""Centralised configuration via Pydantic BaseSettings.

All runtime knobs live here. Values are read from environment variables
(or a .env file) so the same image runs identically in Docker, CI, and
production without code changes.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SensorFlow runtime configuration.

    Every field maps 1-to-1 to an environment variable of the same name
    (case-insensitive). Provide overrides via a .env file at the project root
    or by exporting variables in the shell / CI environment.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #

    project_root: Path = Field(
        default=Path(__file__).resolve().parents[1],
        description="Absolute path to the repository root.",
    )

    data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory where raw SMAP files are stored after ingestion.",
    )

    models_dir: Path = Field(
        default=Path("models"),
        description="Directory where serialised model artefacts are written.",
    )

    reports_dir: Path = Field(
        default=Path("reports"),
        description="Directory where Evidently HTML drift reports are written.",
    )

    # ------------------------------------------------------------------ #
    # SMAP dataset                                                         #
    # ------------------------------------------------------------------ #

    smap_base_url: str = Field(
        default="https://s3-us-west-2.amazonaws.com/telemanom/data",
        description=(
            "S3 base URL for the telemanom dataset. Individual channel arrays are "
            "fetched as {base_url}/train/{chan_id}.npy and {base_url}/test/{chan_id}.npy."
        ),
    )

    smap_labels_url: str = Field(
        default=(
            "https://raw.githubusercontent.com/khundman/telemanom/"
            "master/labeled_anomalies.csv"
        ),
        description="URL to the labeled anomalies CSV.",
    )

    smap_spacecraft: str = Field(
        default="SMAP",
        description=(
            "Spacecraft filter applied when reading labeled_anomalies.csv. "
            "Use 'SMAP' for satellite data only, or leave blank to include MSL as well."
        ),
    )

    smap_n_input_features: int = Field(
        default=25,
        ge=1,
        description="Number of input telemetry features per channel (columns in each .npy array).",
    )

    # ------------------------------------------------------------------ #
    # Feature engineering                                                  #
    # ------------------------------------------------------------------ #

    window_size: int = Field(
        default=64,
        ge=2,
        description="Sliding window length (time-steps) used to build sequences.",
    )

    step_size: int = Field(
        default=1,
        ge=1,
        description="Stride between consecutive windows.",
    )

    n_lag_features: int = Field(
        default=5,
        ge=0,
        description="Number of lag features appended per channel.",
    )

    test_split_ratio: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Fraction of training data held out for local validation.",
    )

    random_seed: int = Field(
        default=42,
        description="Global random seed for reproducibility.",
    )

    # ------------------------------------------------------------------ #
    # Isolation Forest                                                     #
    # ------------------------------------------------------------------ #

    if_n_estimators: int = Field(
        default=200,
        ge=10,
        description="Number of trees in the Isolation Forest ensemble.",
    )

    if_contamination: float = Field(
        default=0.05,
        gt=0.0,
        lt=0.5,
        description="Expected fraction of anomalies — used to set the IF decision threshold.",
    )

    if_max_samples: int | Literal["auto"] = Field(
        default="auto",
        description="Samples drawn per tree ('auto' = min(256, n_samples)).",
    )

    # ------------------------------------------------------------------ #
    # LSTM Autoencoder                                                     #
    # ------------------------------------------------------------------ #

    lstm_hidden_size: int = Field(
        default=64,
        ge=8,
        description="Hidden state dimensionality of LSTM encoder/decoder cells.",
    )

    lstm_num_layers: int = Field(
        default=2,
        ge=1,
        description="Number of stacked LSTM layers.",
    )

    lstm_epochs: int = Field(
        default=30,
        ge=1,
        description="Maximum training epochs for the LSTM autoencoder.",
    )

    lstm_batch_size: int = Field(
        default=256,
        ge=1,
        description="Mini-batch size for LSTM training.",
    )

    lstm_learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Adam optimiser learning rate.",
    )

    lstm_patience: int = Field(
        default=5,
        ge=1,
        description="Early-stopping patience (epochs without val-loss improvement).",
    )

    lstm_max_train_windows: int = Field(
        default=200_000,
        ge=1000,
        description=(
            "Maximum training windows fed to the LSTM per run. "
            "Randomly subsampled when the dataset is larger. "
            "Set to a very large number to use all windows."
        ),
    )

    lstm_reconstruction_threshold_percentile: float = Field(
        default=95.0,
        ge=50.0,
        le=99.9,
        description=(
            "Percentile of training reconstruction errors used to derive "
            "the anomaly decision threshold."
        ),
    )

    # ------------------------------------------------------------------ #
    # MLflow                                                               #
    # ------------------------------------------------------------------ #

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="URI of the MLflow tracking server.",
    )

    mlflow_experiment_name: str = Field(
        default="sensorflow-anomaly-detection",
        description="MLflow experiment that all training runs are logged under.",
    )

    mlflow_model_name: str = Field(
        default="sensorflow-anomaly-detector",
        description="Registered model name in the MLflow Model Registry.",
    )

    mlflow_champion_alias: str = Field(
        default="champion",
        description="Model alias assigned to the production model version.",
    )

    mlflow_challenger_alias: str = Field(
        default="challenger",
        description="Model alias assigned to the candidate model version awaiting promotion.",
    )

    # ------------------------------------------------------------------ #
    # FastAPI                                                              #
    # ------------------------------------------------------------------ #

    api_host: str = Field(
        default="0.0.0.0",
        description="Host address the FastAPI server binds to.",
    )

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port the FastAPI server listens on.",
    )

    api_title: str = Field(
        default="SensorFlow Anomaly Detection API",
        description="Title shown in the OpenAPI docs.",
    )

    api_version: str = Field(
        default="0.1.0",
        description="Semantic version string surfaced by /health.",
    )

    api_workers: int = Field(
        default=1,
        ge=1,
        description="Number of uvicorn worker processes.",
    )

    # ------------------------------------------------------------------ #
    # Drift monitoring & retraining                                        #
    # ------------------------------------------------------------------ #

    drift_threshold: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of channels that must show drift before a retraining "
            "flag is raised."
        ),
    )

    drift_p_value: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Statistical significance threshold for the per-channel drift test.",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                           #
    # ------------------------------------------------------------------ #

    @field_validator("data_dir", "models_dir", "reports_dir", mode="before")
    @classmethod
    def _resolve_relative_paths(cls, v: Path | str) -> Path:
        """Resolve relative paths against the working directory at parse time."""
        return Path(v)


def get_settings() -> Settings:
    """Return a cached Settings instance.

    Call this function everywhere rather than instantiating Settings directly
    so that test suites can monkey-patch it in a single place.
    """
    return _settings


# Module-level singleton — instantiated once at import time.
_settings: Settings = Settings()
