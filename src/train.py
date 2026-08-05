"""Model training with full MLflow tracking and champion/challenger promotion.

Current models
--------------
- Isolation Forest  (sklearn)  — trained on flat lag-feature arrays
- LSTM Autoencoder  (PyTorch)  — trained on sliding-window arrays  [step 4b]

Champion/challenger logic
-------------------------
After training, the new model is registered in the MLflow Model Registry as
``challenger``.  If its validation anomaly score separates normal from
anomalous better than the current ``champion`` (measured by the normalised
separation score), it is promoted to ``champion`` and the previous champion
is demoted to ``challenger``.

When no champion exists yet the first trained model is promoted immediately.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from sklearn.ensemble import IsolationForest

from config import Settings, get_settings

log = logging.getLogger(__name__)

# Tag key used on every MLflow run to identify the model type.
TAG_MODEL_TYPE = "model_type"
# Metric logged to the registry version so champion/challenger can be compared.
METRIC_SEP_SCORE = "val_separation_score"


# ── validation score ──────────────────────────────────────────────────────────


def separation_score(scores: np.ndarray, labels: np.ndarray | None) -> float:
    """Compute how well anomaly scores separate normal from anomalous points.

    Uses the normalised difference between the mean anomaly score of anomalous
    points and the mean score of normal points, divided by the pooled std.
    Falls back to the mean score (negated, so higher = more anomalous) when no
    labels are available.

    Higher is better — a score near 1.0 means clean separation.

    Args:
        scores: Raw decision scores from the model (higher = more anomalous).
        labels: Optional boolean array; ``True`` indicates anomalous.

    Returns:
        Float separation score in the range ``(-inf, inf)``.
    """
    if labels is None or not labels.any():
        return float(-scores.mean())

    normal = scores[~labels]
    anomalous = scores[labels]
    pooled_std = float(np.sqrt((normal.var() + anomalous.var()) / 2 + 1e-8))
    return float((anomalous.mean() - normal.mean()) / pooled_std)


# ── MLflow helpers ────────────────────────────────────────────────────────────


def _get_or_create_experiment(name: str) -> str:
    """Return the MLflow experiment ID, creating it if it does not exist."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name)


def _current_champion_score(client: MlflowClient, model_name: str, alias: str) -> float | None:
    """Return the separation score of the current champion, or None if absent."""
    try:
        version: ModelVersion = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(version.run_id)
        return float(run.data.metrics[METRIC_SEP_SCORE])
    except Exception:
        return None


def _register_and_tag(
    run_id: str,
    model_uri: str,
    model_name: str,
    alias: str,
    client: MlflowClient,
) -> ModelVersion:
    """Register a model version and assign *alias* to it."""
    version: ModelVersion = mlflow.register_model(model_uri, model_name)
    client.set_registered_model_alias(model_name, alias, version.version)
    log.info("registered %s v%s as '%s'", model_name, version.version, alias)
    return version


# ── champion / challenger promotion ──────────────────────────────────────────


def maybe_promote_champion(
    run_id: str,
    new_score: float,
    model_uri: str,
    cfg: Settings,
) -> str:
    """Promote the new model to champion if it beats the current one.

    Args:
        run_id: MLflow run ID of the newly trained model.
        new_score: Separation score of the new model on the validation set.
        model_uri: ``runs:/<run_id>/model`` URI of the trained artefact.
        cfg: Settings instance.

    Returns:
        The alias assigned to the new model version
        (``champion`` or ``challenger``).
    """
    client = MlflowClient()
    current_score = _current_champion_score(client, cfg.mlflow_model_name, cfg.mlflow_champion_alias)

    if current_score is None or new_score > current_score:
        # Demote the old champion to challenger before promoting the new one
        # so the registry never has two champions simultaneously.
        if current_score is not None:
            old_version = client.get_model_version_by_alias(
                cfg.mlflow_model_name, cfg.mlflow_champion_alias
            )
            client.set_registered_model_alias(
                cfg.mlflow_model_name, cfg.mlflow_challenger_alias, old_version.version
            )
            log.info("demoted v%s to '%s'", old_version.version, cfg.mlflow_challenger_alias)

        _register_and_tag(run_id, model_uri, cfg.mlflow_model_name, cfg.mlflow_champion_alias, client)
        log.info("new champion — score %.4f > previous %.4f", new_score, current_score or 0.0)
        return cfg.mlflow_champion_alias

    _register_and_tag(run_id, model_uri, cfg.mlflow_model_name, cfg.mlflow_challenger_alias, client)
    log.info(
        "challenger retained — score %.4f ≤ champion %.4f", new_score, current_score
    )
    return cfg.mlflow_challenger_alias


# ── Isolation Forest ──────────────────────────────────────────────────────────


def train_isolation_forest(
    train_flat: np.ndarray,
    val_flat: np.ndarray,
    val_labels: np.ndarray | None,
    cfg: Settings,
) -> tuple[IsolationForest, float]:
    """Fit an Isolation Forest and return the model with its validation score.

    Args:
        train_flat: Shape ``(n_train, n_features)``.
        val_flat: Shape ``(n_val, n_features)``.
        val_labels: Optional boolean anomaly labels for *val_flat*.
        cfg: Settings instance.

    Returns:
        Tuple of ``(fitted_model, separation_score)``.
    """
    log.info(
        "training Isolation Forest — n_estimators=%d contamination=%.3f",
        cfg.if_n_estimators,
        cfg.if_contamination,
    )
    t0 = time.perf_counter()

    model = IsolationForest(
        n_estimators=cfg.if_n_estimators,
        contamination=cfg.if_contamination,
        max_samples=cfg.if_max_samples,
        random_state=cfg.random_seed,
        n_jobs=-1,
    )
    model.fit(train_flat)

    elapsed = time.perf_counter() - t0
    log.info("training complete in %.1fs", elapsed)

    # decision_function returns higher scores for normal, lower for anomalous.
    # We negate so "higher = more anomalous" everywhere in this codebase.
    val_scores = -model.decision_function(val_flat)
    score = separation_score(val_scores, val_labels)
    log.info("val separation score: %.4f", score)

    return model, score


def run_isolation_forest(cfg: Settings | None = None) -> dict[str, Any]:
    """Train an Isolation Forest, log to MLflow, and promote if champion.

    Args:
        cfg: Settings instance; uses the module singleton when ``None``.

    Returns:
        Dict with keys ``run_id``, ``alias``, ``val_separation_score``.
    """
    if cfg is None:
        cfg = get_settings()

    features_dir = cfg.data_dir.parent / "features"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    train_flat: np.ndarray = np.load(features_dir / "train_flat.npy")
    val_flat: np.ndarray = np.load(features_dir / "val_flat.npy")

    val_labels: np.ndarray | None = None
    val_labels_path = features_dir / "val_flat_labels.npy"
    if val_labels_path.exists():
        val_labels = np.load(val_labels_path)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    experiment_id = _get_or_create_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tag(TAG_MODEL_TYPE, "isolation_forest")
        mlflow.set_tag("smap_spacecraft", cfg.smap_spacecraft)

        mlflow.log_params(
            {
                "n_estimators": cfg.if_n_estimators,
                "contamination": cfg.if_contamination,
                "max_samples": cfg.if_max_samples,
                "n_lag_features": cfg.n_lag_features,
                "window_size": cfg.window_size,
                "random_seed": cfg.random_seed,
                "train_rows": train_flat.shape[0],
                "n_input_features": train_flat.shape[1],
            }
        )

        model, val_score = train_isolation_forest(train_flat, val_flat, val_labels, cfg)

        mlflow.log_metric(METRIC_SEP_SCORE, val_score)

        # Score distribution on val for debugging
        val_scores = -model.decision_function(val_flat)
        mlflow.log_metrics(
            {
                "val_score_mean": float(val_scores.mean()),
                "val_score_std": float(val_scores.std()),
                "val_score_p95": float(np.percentile(val_scores, 95)),
            }
        )

        mlflow.sklearn.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"

        # Persist locally as well for fast offline loading
        local_path = cfg.models_dir / "isolation_forest.joblib"
        joblib.dump(model, local_path)
        log.info("model saved locally → %s", local_path)

        run_id: str = run.info.run_id

    alias = maybe_promote_champion(run_id, val_score, model_uri, cfg)

    return {
        "run_id": run_id,
        "alias": alias,
        "val_separation_score": val_score,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_isolation_forest()
    log.info("done: %s", result)
