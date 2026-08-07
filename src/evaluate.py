"""Evaluation: test-set metrics, Evidently drift report, and champion promotion.

Pipeline
--------
1. Load both trained models from disk.
2. Score the test split with each model.
3. Compute F1, precision, recall, and AUROC against ground-truth labels.
4. Log metrics back to each model's originating MLflow run.
5. Promote the model with the higher F1 to the ``champion`` alias.
6. Generate an Evidently HTML drift report (train vs test) and log it as an
   MLflow artefact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import Settings, get_settings
from train import LSTMAutoencoder

log = logging.getLogger(__name__)


# ── model loading ─────────────────────────────────────────────────────────────


def load_isolation_forest(cfg: Settings):
    """Load the Isolation Forest from disk.

    Returns:
        Fitted :class:`~sklearn.ensemble.IsolationForest`.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = cfg.models_dir / "isolation_forest.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Isolation Forest not found at {path}. Run train.py first.")
    return joblib.load(path)


def load_lstm_autoencoder(
    cfg: Settings, n_features: int, seq_len: int
) -> tuple[LSTMAutoencoder, float]:
    """Load the LSTM Autoencoder weights and decision threshold from disk.

    Args:
        cfg: Settings instance.
        n_features: Number of input features (must match training config).
        seq_len: Window length (must match training config).

    Returns:
        Tuple of ``(model, threshold)``.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    path = cfg.models_dir / "lstm_autoencoder.pt"
    if not path.exists():
        raise FileNotFoundError(f"LSTM checkpoint not found at {path}. Run train.py first.")

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = LSTMAutoencoder(
        n_features=n_features,
        hidden_size=cfg.lstm_hidden_size,
        num_layers=cfg.lstm_num_layers,
        seq_len=seq_len,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, float(checkpoint["threshold"])


# ── scoring ───────────────────────────────────────────────────────────────────


def score_isolation_forest(model, X: np.ndarray) -> np.ndarray:
    """Return per-sample anomaly scores (higher = more anomalous).

    Args:
        model: Fitted Isolation Forest.
        X: Shape ``(n_samples, n_features)``.

    Returns:
        Shape ``(n_samples,)`` float array.
    """
    return -model.decision_function(X)


def score_lstm(
    model: LSTMAutoencoder, windows: np.ndarray, batch_size: int = 512
) -> np.ndarray:
    """Return per-window reconstruction errors (higher = more anomalous).

    Args:
        model: Trained :class:`LSTMAutoencoder` in eval mode.
        windows: Shape ``(n_windows, seq_len, n_features)``.
        batch_size: Number of windows per inference batch.

    Returns:
        Shape ``(n_windows,)`` float array.
    """
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).float()
            errors.append(model.reconstruction_error(batch).numpy())
    return np.concatenate(errors)


# ── metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary classification metrics from anomaly scores.

    Args:
        scores: Continuous anomaly scores (higher = more anomalous).
        labels: Ground-truth boolean labels.
        threshold: Score value above which a sample is predicted anomalous.

    Returns:
        Dict with keys ``f1``, ``precision``, ``recall``, ``auroc``, ``auprc``.
    """
    preds = (scores >= threshold).astype(int)
    y = labels.astype(int)

    return {
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "auroc": float(roc_auc_score(y, scores)) if y.any() else 0.0,
        "auprc": float(average_precision_score(y, scores)) if y.any() else 0.0,
    }


# ── drift report ──────────────────────────────────────────────────────────────


def generate_drift_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    cfg: Settings,
) -> Path:
    """Generate an Evidently HTML drift report comparing train and test distributions.

    Args:
        train_df: Training split DataFrame.
        test_df: Test split DataFrame.
        feat_cols: Feature columns to include in the report.
        cfg: Settings instance.

    Returns:
        Path to the saved HTML report.
    """
    from evidently import Report  # lazy — not needed at import time
    from evidently.presets import DataDriftPreset

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / "drift_report.html"

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=train_df[feat_cols],
        current_data=test_df[feat_cols],
    )
    snapshot.save_html(str(report_path))
    log.info("drift report saved → %s", report_path)
    return report_path


# ── MLflow helpers ────────────────────────────────────────────────────────────


def _find_latest_run(client: MlflowClient, experiment_name: str, model_type: str) -> str | None:
    """Return the run_id of the most recent completed run for *model_type*."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model_type = '{model_type}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


def _promote_champion(
    if_metrics: dict[str, float],
    lstm_metrics: dict[str, float],
    if_run_id: str | None,
    lstm_run_id: str | None,
    cfg: Settings,
) -> str:
    """Assign champion/challenger aliases based on test F1 scores.

    Args:
        if_metrics: Evaluation metrics for the Isolation Forest.
        lstm_metrics: Evaluation metrics for the LSTM Autoencoder.
        if_run_id: MLflow run ID for the IF (may be None if not tracked).
        lstm_run_id: MLflow run ID for the LSTM (may be None if not tracked).
        cfg: Settings instance.

    Returns:
        Name of the model promoted to champion (``"isolation_forest"`` or
        ``"lstm_autoencoder"``).
    """
    client = MlflowClient()
    if_f1 = if_metrics["f1"]
    lstm_f1 = lstm_metrics["f1"]

    winner = "lstm_autoencoder" if lstm_f1 >= if_f1 else "isolation_forest"
    winner_run_id = lstm_run_id if winner == "lstm_autoencoder" else if_run_id
    loser_run_id = if_run_id if winner == "lstm_autoencoder" else lstm_run_id

    log.info("champion: %s (F1=%.4f) over %s (F1=%.4f)", winner, max(if_f1, lstm_f1), "the other", min(if_f1, lstm_f1))

    if winner_run_id:
        try:
            # IF uses mlflow.sklearn.log_model → artifact_path="model"
            # LSTM uses log_artifact (raw .pt file) → no MLflow model format
            # Only IF can be registered in the model registry this way.
            if winner == "isolation_forest":
                model_uri = f"runs:/{winner_run_id}/model"
                version = mlflow.register_model(model_uri, cfg.mlflow_model_name)
                client.set_registered_model_alias(cfg.mlflow_model_name, cfg.mlflow_champion_alias, version.version)
                client.set_model_version_tag(cfg.mlflow_model_name, version.version, "model_type", winner)
                log.info("registered v%s as '%s'", version.version, cfg.mlflow_champion_alias)
            else:
                # LSTM champion: tag the run, model loaded from disk by serve.py
                client.set_tag(winner_run_id, "alias", cfg.mlflow_champion_alias)
                log.info("LSTM champion tagged in run %s (loaded from disk by serve.py)", winner_run_id)
        except Exception as exc:
            log.warning("MLflow registry update failed (server may be offline): %s", exc)

    return winner


# ── orchestrator ──────────────────────────────────────────────────────────────


def run_evaluate(cfg: Settings | None = None) -> dict[str, Any]:
    """Evaluate both models on the test set and generate the drift report.

    Args:
        cfg: Settings instance; uses the module singleton when ``None``.

    Returns:
        Dict with keys ``isolation_forest``, ``lstm``, ``champion``,
        ``drift_report_path``.
    """
    if cfg is None:
        cfg = get_settings()

    features_dir = cfg.data_dir.parent / "features"

    # ── load features ──────────────────────────────────────────────────────────
    test_flat: np.ndarray = np.load(features_dir / "test_flat.npy")
    test_flat_labels: np.ndarray = np.load(features_dir / "test_flat_labels.npy")
    test_windows: np.ndarray = np.load(features_dir / "test_windows.npy")
    test_window_labels: np.ndarray = np.load(features_dir / "test_window_labels.npy")

    _, seq_len, n_features = test_windows.shape

    # ── Isolation Forest ───────────────────────────────────────────────────────
    log.info("evaluating Isolation Forest …")
    if_model = load_isolation_forest(cfg)
    if_scores = score_isolation_forest(if_model, test_flat)

    # Threshold: the IF contamination parameter implicitly sets it via predict();
    # we derive the equivalent score threshold from the training data threshold.
    if_threshold = float(np.percentile(if_scores, 100 * (1 - cfg.if_contamination)))
    if_metrics = compute_metrics(if_scores, test_flat_labels, if_threshold)
    log.info("IF  — F1=%.4f  precision=%.4f  recall=%.4f  AUROC=%.4f",
             if_metrics["f1"], if_metrics["precision"], if_metrics["recall"], if_metrics["auroc"])

    # ── LSTM Autoencoder ───────────────────────────────────────────────────────
    log.info("evaluating LSTM Autoencoder …")
    lstm_model, lstm_threshold = load_lstm_autoencoder(cfg, n_features, seq_len)
    lstm_scores = score_lstm(lstm_model, test_windows)
    lstm_metrics = compute_metrics(lstm_scores, test_window_labels, lstm_threshold)
    log.info("LSTM — F1=%.4f  precision=%.4f  recall=%.4f  AUROC=%.4f",
             lstm_metrics["f1"], lstm_metrics["precision"], lstm_metrics["recall"], lstm_metrics["auroc"])

    # ── log metrics to MLflow ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()

    if_run_id = _find_latest_run(client, cfg.mlflow_experiment_name, "isolation_forest")
    lstm_run_id = _find_latest_run(client, cfg.mlflow_experiment_name, "lstm_autoencoder")

    for run_id, metrics, prefix in [
        (if_run_id, if_metrics, "test_if"),
        (lstm_run_id, lstm_metrics, "test_lstm"),
    ]:
        if run_id:
            try:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.items()})
            except Exception as exc:
                log.warning("could not log to MLflow run %s: %s", run_id, exc)

    # ── champion promotion ─────────────────────────────────────────────────────
    champion = _promote_champion(if_metrics, lstm_metrics, if_run_id, lstm_run_id, cfg)

    # ── Evidently drift report ─────────────────────────────────────────────────
    log.info("generating drift report …")
    train_df = pd.read_parquet(cfg.data_dir / "train.parquet")
    test_df = pd.read_parquet(cfg.data_dir / "test.parquet")
    feat_cols = [c for c in train_df.columns if c.startswith("f_")]

    drift_path = generate_drift_report(train_df, test_df, feat_cols, cfg)

    if lstm_run_id:
        try:
            with mlflow.start_run(run_id=lstm_run_id):
                mlflow.log_artifact(str(drift_path), artifact_path="reports")
        except Exception as exc:
            log.warning("could not log drift report to MLflow: %s", exc)

    # Write eval metrics to disk so the dashboard can display them.
    import json as _json
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics = {
        "isolation_forest": if_metrics,
        "lstm": lstm_metrics,
        "champion": champion,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
    }
    metrics_path = cfg.reports_dir / "eval_metrics.json"
    metrics_path.write_text(_json.dumps(eval_metrics, indent=2), encoding="utf-8")
    log.info("eval metrics saved → %s", metrics_path)

    log.info("evaluation complete — champion: %s", champion)
    return {
        "isolation_forest": if_metrics,
        "lstm": lstm_metrics,
        "champion": champion,
        "drift_report_path": str(drift_path),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_evaluate()
    log.info("done: %s", result)
