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
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import torch
import torch.nn as nn
from config import Settings, get_settings
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


# ── LSTM Autoencoder ──────────────────────────────────────────────────────────


class LSTMEncoder(nn.Module):
    """Encodes a window of shape ``(batch, seq_len, n_features)`` into a
    hidden state of shape ``(batch, hidden_size)``."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only the final hidden state is needed for the decoder seed.
        _, (h_n, _) = self.lstm(x)
        return h_n  # (num_layers, batch, hidden_size)


class LSTMDecoder(nn.Module):
    """Reconstructs a sequence from the encoder's final hidden state."""

    def __init__(
        self, n_features: int, hidden_size: int, num_layers: int, seq_len: int
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, h_n: torch.Tensor) -> torch.Tensor:
        batch = h_n.shape[1]
        # Start token: zeros of shape (batch, 1, n_features) — the decoder
        # generates each step conditioned only on the encoder's hidden state.
        decoder_input = torch.zeros(
            batch, self.seq_len, self.output_layer.out_features, device=h_n.device
        )
        c_n = torch.zeros_like(h_n)
        out, _ = self.lstm(decoder_input, (h_n, c_n))
        return self.output_layer(out)  # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    """Sequence-to-sequence autoencoder for anomaly detection via reconstruction error."""

    def __init__(
        self, n_features: int, hidden_size: int, num_layers: int, seq_len: int
    ) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers)
        self.decoder = LSTMDecoder(n_features, hidden_size, num_layers, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_n = self.encoder(x)
        return self.decoder(h_n)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample MSE between input and reconstruction.

        Args:
            x: Shape ``(batch, seq_len, n_features)``.

        Returns:
            Shape ``(batch,)`` — one scalar error per window.
        """
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))


def _make_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_lstm_autoencoder(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray | None,
    cfg: Settings,
) -> tuple[LSTMAutoencoder, float, float]:
    """Train an LSTM Autoencoder with early stopping.

    Args:
        train_windows: Shape ``(n_train, window_size, n_features)``.
        val_windows: Shape ``(n_val, window_size, n_features)``.
        val_labels: Optional boolean anomaly labels aligned to *val_windows*.
        cfg: Settings instance.

    Returns:
        Tuple of ``(model, threshold, val_separation_score)`` where *threshold*
        is the 95th-percentile reconstruction error on training data — used at
        inference time to decide anomaly/normal.
    """
    device = _make_device()
    _, seq_len, n_features = train_windows.shape

    # Subsample training windows to keep each epoch tractable on CPU.
    if len(train_windows) > cfg.lstm_max_train_windows:
        rng = np.random.default_rng(cfg.random_seed)
        idx = rng.choice(len(train_windows), size=cfg.lstm_max_train_windows, replace=False)
        train_windows = train_windows[idx]
        log.info("subsampled train windows: %d → %d", len(idx), cfg.lstm_max_train_windows)

    log.info(
        "training LSTM Autoencoder — hidden=%d layers=%d epochs=%d device=%s windows=%d",
        cfg.lstm_hidden_size,
        cfg.lstm_num_layers,
        cfg.lstm_epochs,
        device,
        len(train_windows),
    )

    model = LSTMAutoencoder(n_features, cfg.lstm_hidden_size, cfg.lstm_num_layers, seq_len).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.lstm_learning_rate)
    criterion = nn.MSELoss()

    train_tensor = torch.from_numpy(train_windows).float().to(device)
    val_tensor = torch.from_numpy(val_windows).float().to(device)

    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=cfg.lstm_batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=cfg.lstm_batch_size,
        shuffle=False,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict[str, torch.Tensor] = {}
    start_epoch = 1

    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if one exists.
    ckpt_path = cfg.models_dir / "lstm_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimiser.load_state_dict(ckpt["optimiser_state"])
        best_val_loss = ckpt["best_val_loss"]
        best_state = ckpt["best_state"]
        patience_counter = ckpt["patience_counter"]
        start_epoch = ckpt["epoch"] + 1
        log.info("resumed from checkpoint — epoch %d, best_val_loss=%.5f", ckpt["epoch"], best_val_loss)

    n_batches = len(train_loader)
    log.info("batches per epoch: %d  (batch_size=%d)", n_batches, cfg.lstm_batch_size)

    for epoch in range(start_epoch, cfg.lstm_epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.lstm_epochs}", unit="batch", leave=False)
        for (batch,) in pbar:
            optimiser.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
            pbar.set_postfix(loss=f"{loss.item():.5f}")
        epoch_loss /= len(train_windows)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(b), b).item() * len(b) for (b,) in val_loader
            ) / len(val_tensor)

        log.info("epoch %d/%d — train_loss=%.5f", epoch, cfg.lstm_epochs, epoch_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg.lstm_patience:
                log.info("early stopping at epoch %d", epoch)
                # Save final checkpoint before breaking.
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimiser_state": optimiser.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_state": best_state,
                    "patience_counter": patience_counter,
                }, ckpt_path)
                break

        # Save checkpoint after every epoch.
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict(),
            "best_val_loss": best_val_loss,
            "best_state": best_state,
            "patience_counter": patience_counter,
        }, ckpt_path)
        log.info("checkpoint saved → %s", ckpt_path)

    model.load_state_dict(best_state)
    model.eval()

    # Reconstruction error threshold: 95th percentile on training data.
    # Batched to avoid OOM on large datasets.
    train_error_loader = DataLoader(
        TensorDataset(train_tensor), batch_size=cfg.lstm_batch_size, shuffle=False
    )
    with torch.no_grad():
        train_errors = np.concatenate([
            model.reconstruction_error(b).cpu().numpy() for (b,) in train_error_loader
        ])
    threshold = float(np.percentile(train_errors, cfg.lstm_reconstruction_threshold_percentile))
    log.info("anomaly threshold (p%.0f): %.6f", cfg.lstm_reconstruction_threshold_percentile, threshold)

    # Validation separation score.
    with torch.no_grad():
        val_errors = np.concatenate([
            model.reconstruction_error(b).cpu().numpy() for (b,) in val_loader
        ])
    score = separation_score(val_errors, val_labels)
    log.info("val separation score: %.4f  val_loss: %.5f", score, best_val_loss)

    return model, threshold, score


def run_lstm_autoencoder(cfg: Settings | None = None) -> dict[str, Any]:
    """Train an LSTM Autoencoder, log to MLflow, and promote if champion.

    Args:
        cfg: Settings instance; uses the module singleton when ``None``.

    Returns:
        Dict with keys ``run_id``, ``alias``, ``val_separation_score``,
        ``threshold``.
    """
    if cfg is None:
        cfg = get_settings()

    features_dir = cfg.data_dir.parent / "features"
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    train_windows: np.ndarray = np.load(features_dir / "train_windows.npy")
    val_windows: np.ndarray = np.load(features_dir / "val_windows.npy")

    val_labels: np.ndarray | None = None
    val_labels_path = features_dir / "val_window_labels.npy"
    if val_labels_path.exists():
        val_labels = np.load(val_labels_path)

    _, seq_len, n_features = train_windows.shape

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    experiment_id = _get_or_create_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tag(TAG_MODEL_TYPE, "lstm_autoencoder")
        mlflow.set_tag("smap_spacecraft", cfg.smap_spacecraft)

        mlflow.log_params(
            {
                "hidden_size": cfg.lstm_hidden_size,
                "num_layers": cfg.lstm_num_layers,
                "epochs": cfg.lstm_epochs,
                "batch_size": cfg.lstm_batch_size,
                "learning_rate": cfg.lstm_learning_rate,
                "patience": cfg.lstm_patience,
                "threshold_percentile": cfg.lstm_reconstruction_threshold_percentile,
                "window_size": seq_len,
                "n_input_features": n_features,
                "train_windows": train_windows.shape[0],
                "random_seed": cfg.random_seed,
            }
        )

        model, threshold, val_score = train_lstm_autoencoder(
            train_windows, val_windows, val_labels, cfg
        )

        mlflow.log_metric(METRIC_SEP_SCORE, val_score)
        mlflow.log_metric("reconstruction_threshold", threshold)

        # Save model weights + threshold to disk (serve.py loads from here).
        model_path = cfg.models_dir / "lstm_autoencoder.pt"
        torch.save({"state_dict": model.state_dict(), "threshold": threshold}, model_path)
        mlflow.log_artifact(str(model_path))
        log.info("model saved locally → %s", model_path)

        run_id: str = run.info.run_id

        # Determine champion/challenger by comparing val scores.
        # LSTM artefact is disk-based so we skip the MLflow model registry
        # and record the alias as a run tag instead.
        client = MlflowClient()
        current_score = _current_champion_score(client, cfg.mlflow_model_name, cfg.mlflow_champion_alias)
        if current_score is None or val_score > current_score:
            alias = cfg.mlflow_champion_alias
            log.info("new champion — score %.4f > previous %.4f", val_score, current_score or 0.0)
        else:
            alias = cfg.mlflow_challenger_alias
            log.info("challenger retained — score %.4f ≤ champion %.4f", val_score, current_score)
        mlflow.set_tag("alias", alias)
        mlflow.set_tag("model_path", str(model_path))

    # Only delete checkpoint after MLflow logging is fully complete.
    ckpt_path = cfg.models_dir / "lstm_checkpoint.pt"
    if ckpt_path.exists():
        ckpt_path.unlink()
        log.info("checkpoint removed — all artefacts saved successfully")

    return {
        "run_id": run_id,
        "alias": alias,
        "val_separation_score": val_score,
        "threshold": threshold,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run_isolation_forest()
    log.info("IF done: %s", result)
    result = run_lstm_autoencoder()
    log.info("LSTM done: %s", result)
