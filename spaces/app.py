"""SensorFlow — self-contained Streamlit app for HuggingFace Spaces.

Loads trained models directly from disk (no FastAPI required).
Channels are generated on-the-fly using the same AR(1) synthetic process
used during training; anomaly labels come from the real labeled_anomalies.csv.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
LABELS_CSV = ROOT / "labeled_anomalies.csv"
REPORTS_DIR = ROOT / "reports"

ANOMALY_COLOR = "#ef4444"
NORMAL_COLOR = "#3b82f6"
THRESHOLD_COLOR = "#f59e0b"

N_FEATURES = 25
WINDOW_SIZE = 64
N_LAG_FEATURES = 5

# ── model architecture (mirrors train.py) ─────────────────────────────────────


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n


class LSTMDecoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, h_n: torch.Tensor) -> torch.Tensor:
        batch = h_n.shape[1]
        decoder_input = torch.zeros(batch, self.seq_len, self.output_layer.out_features)
        c_n = torch.zeros_like(h_n)
        out, _ = self.lstm(decoder_input, (h_n, c_n))
        return self.output_layer(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, seq_len: int) -> None:
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_size, num_layers)
        self.decoder = LSTMDecoder(n_features, hidden_size, num_layers, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))


# ── synthetic channel generation ──────────────────────────────────────────────


def generate_channel(
    chan_id: str,
    n_timesteps: int,
    anomaly_sequences: list[list[int]],
) -> np.ndarray:
    seed = int.from_bytes(chan_id.encode(), "little") % (2**31)
    rng = np.random.default_rng(seed + 1)  # +1 → test split differs from train
    phi, sigma = 0.85, 0.3
    arr = np.zeros((n_timesteps, N_FEATURES), dtype=np.float32)
    arr[0] = (rng.standard_normal(N_FEATURES) * sigma).astype(np.float32)
    for t in range(1, n_timesteps):
        arr[t] = (phi * arr[t - 1] + rng.standard_normal(N_FEATURES) * sigma).astype(np.float32)
    n_anom_feats = max(1, N_FEATURES // 3)
    for start, end in anomaly_sequences:
        if start >= n_timesteps:
            continue
        end = min(end, n_timesteps - 1)
        feats = rng.choice(N_FEATURES, size=n_anom_feats, replace=False)
        arr[start:end + 1, feats] += rng.choice([-1, 1]) * 3.0
    return arr


# ── model loading ─────────────────────────────────────────────────────────────


@dataclass
class ModelState:
    model_type: str
    model: object
    scaler: StandardScaler
    threshold: float


@st.cache_resource
def load_scaler() -> StandardScaler | None:
    path = MODELS_DIR / "scaler.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_resource
def load_lstm() -> ModelState | None:
    scaler = load_scaler()
    path = MODELS_DIR / "lstm_autoencoder.pt"
    if scaler is None or not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = LSTMAutoencoder(N_FEATURES, hidden_size=64, num_layers=2, seq_len=WINDOW_SIZE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return ModelState("lstm_autoencoder", model, scaler, float(ckpt["threshold"]))


@st.cache_resource
def load_isolation_forest() -> ModelState | None:
    scaler = load_scaler()
    path = MODELS_DIR / "isolation_forest.joblib"
    if scaler is None or not path.exists():
        return None
    return ModelState("isolation_forest", joblib.load(path), scaler, 0.0)


def available_models() -> dict[str, ModelState]:
    out = {}
    if (m := load_lstm()) is not None:
        out["LSTM Autoencoder"] = m
    if (m := load_isolation_forest()) is not None:
        out["Isolation Forest"] = m
    return out


# ── channel data ──────────────────────────────────────────────────────────────


@st.cache_data
def load_channels() -> dict[str, tuple[np.ndarray, list[list[int]]]]:
    if not LABELS_CSV.exists():
        return {}
    df = pd.read_csv(LABELS_CSV)
    df = df[df["spacecraft"] == "SMAP"]
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    result = {}
    for _, row in df.iterrows():
        result[row["chan_id"]] = (int(row["num_values"]), row["anomaly_sequences"])
    return result


# ── inference ─────────────────────────────────────────────────────────────────


def add_lag_features(arr: np.ndarray, n_lags: int) -> np.ndarray:
    parts = [arr[n_lags:]]
    for lag in range(1, n_lags + 1):
        parts.append(arr[n_lags - lag: len(arr) - lag])
    return np.concatenate(parts, axis=1)


def sliding_window(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    if n < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, arr.shape[1]), dtype=np.float32)
    return np.stack([arr[i:i + WINDOW_SIZE] for i in range(n - WINDOW_SIZE + 1)])


def predict(state: ModelState, readings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scaled = state.scaler.transform(readings)

    if state.model_type == "lstm_autoencoder":
        windows = sliding_window(scaled.astype(np.float32))
        if len(windows) == 0:
            return np.array([]), np.array([]), state.threshold
        timesteps = np.arange(WINDOW_SIZE - 1, WINDOW_SIZE - 1 + len(windows))
        tensor = torch.from_numpy(windows)
        with torch.no_grad():
            scores = state.model.reconstruction_error(tensor).numpy()
        return timesteps, scores, state.threshold

    else:
        flat = add_lag_features(scaled, N_LAG_FEATURES)
        timesteps = np.arange(N_LAG_FEATURES, N_LAG_FEATURES + len(flat))
        scores = -state.model.decision_function(flat)
        threshold = float(np.percentile(scores, 95))
        return timesteps, scores, threshold


# ── plots ─────────────────────────────────────────────────────────────────────


def score_plot(timesteps, scores, flags, threshold, chan_id):
    normal_x = [int(t) for t, f in zip(timesteps, flags) if not f]
    normal_y = [float(s) for s, f in zip(scores, flags) if not f]
    anom_x = [int(t) for t, f in zip(timesteps, flags) if f]
    anom_y = [float(s) for s, f in zip(scores, flags) if f]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=normal_x, y=normal_y, mode="markers",
                             marker=dict(color=NORMAL_COLOR, size=3, opacity=0.6), name="Normal"))
    fig.add_trace(go.Scatter(x=anom_x, y=anom_y, mode="markers",
                             marker=dict(color=ANOMALY_COLOR, size=6, symbol="x"), name="Anomaly"))
    y_max = max(float(scores.max()) if len(scores) else 1.0, threshold)
    fig.add_hline(y=threshold, line_dash="dash", line_color=THRESHOLD_COLOR,
                  annotation_text=f"threshold={threshold:.4f}", annotation_position="top right")
    fig.update_layout(title=f"Anomaly scores — {chan_id}", xaxis_title="Timestep",
                      yaxis_title="Score", yaxis=dict(range=[0, y_max * 1.1]),
                      height=380, margin=dict(l=40, r=20, t=60, b=40))
    return fig


def signal_plot(readings, timesteps, flags, chan_id):
    anom_set = {int(t) for t, f in zip(timesteps, flags) if f}
    n, _ = readings.shape
    ts = list(range(n))
    fig = go.Figure()
    for i in range(N_FEATURES):
        fig.add_trace(go.Scatter(x=ts, y=readings[:, i].tolist(), mode="lines",
                                 line=dict(width=1), opacity=0.4, showlegend=False))
    if anom_set:
        sorted_a = sorted(anom_set)
        spans, start, prev = [], sorted_a[0], sorted_a[0]
        for t in sorted_a[1:]:
            if t > prev + 1:
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))
        y_min, y_max = float(readings.min()), float(readings.max())
        for s, e in spans:
            fig.add_shape(type="rect", x0=s, x1=e, y0=y_min, y1=y_max,
                          fillcolor=ANOMALY_COLOR, opacity=0.15, line_width=0, layer="below")
    fig.update_layout(title=f"Raw signal — {chan_id}", xaxis_title="Timestep",
                      yaxis_title="Value", height=300, margin=dict(l=40, r=20, t=60, b=40))
    return fig


# ── eval metrics ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_eval_metrics() -> dict | None:
    path = REPORTS_DIR / "eval_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def load_drift_flag() -> dict | None:
    path = REPORTS_DIR / "retrain_flag.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── page ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SensorFlow", page_icon="📡", layout="wide")

models = available_models()
channels = load_channels()

eval_data = load_eval_metrics()
champion_key = eval_data.get("champion", "") if eval_data else ""
champion_label = {"isolation_forest": "Isolation Forest", "lstm": "LSTM Autoencoder"}.get(champion_key, "")

with st.sidebar:
    st.markdown("**Best-performing model**")
    st.caption("Champion selected by highest F1 on the SMAP test set — loaded by default in Predict.")
    if models:
        best_label = champion_label if champion_label in models else next(iter(models))
        best = models[best_label]
        with st.container(border=True):
            st.markdown(best_label)
            st.markdown(f"Anomaly threshold: `{best.threshold:.4f}`")
    else:
        st.error("No model found in models/")
    st.divider()
    st.markdown("**About**")
    st.caption("Anomaly detection pipeline for NASA SMAP satellite telemetry — ingestion, training, evaluation, and drift monitoring.")

# ── page header (always visible) ─────────────────────────────────────────────

st.title("📡 SensorFlow")
st.caption(
    "Anomaly detection for NASA SMAP satellite telemetry — "
    "score channels with an LSTM Autoencoder or Isolation Forest, "
    "review model evaluation metrics, and monitor feature drift."
)

tab_predict, tab_analysis, tab_datasource, tab_arch = st.tabs(["Predict", "Evaluate", "Data Sources", "Architecture"])

# ── predict tab ───────────────────────────────────────────────────────────────

with tab_predict:
    ctrl_col, _ = st.columns([1, 2])
    with ctrl_col:
        if channels:
            chan_id = st.selectbox("Channel", sorted(channels.keys()))
        else:
            st.warning("labeled_anomalies.csv not found.")
            chan_id = None
        model_label = st.selectbox("Model", list(models.keys()), disabled=not models)
        state = models.get(model_label)
        predict_btn = st.button("Predict", disabled=(chan_id is None or state is None),
                                use_container_width=True)

    if predict_btn and chan_id and state:
        n_timesteps, anomaly_seqs = channels[chan_id]
        with st.spinner(f"Generating channel {chan_id} and scoring …"):
            readings = generate_channel(chan_id, n_timesteps, anomaly_seqs)
            timesteps, scores, threshold = predict(state, readings)
            flags = scores >= threshold
            st.session_state["result"] = (chan_id, readings, timesteps, scores, flags, threshold)

    if "result" in st.session_state:
        chan_id_r, readings_r, timesteps_r, scores_r, flags_r, threshold_r = st.session_state["result"]
        n_anom = int(flags_r.sum())
        st.subheader(f"Channel {chan_id_r} — {n_anom} anomalies in {len(scores_r)} scored windows")
        st.plotly_chart(score_plot(timesteps_r, scores_r, flags_r, threshold_r, chan_id_r),
                        use_container_width=True)
        st.plotly_chart(signal_plot(readings_r, timesteps_r, flags_r, chan_id_r),
                        use_container_width=True)
        with st.expander("Raw scores"):
            st.dataframe(pd.DataFrame({
                "timestep": timesteps_r, "score": scores_r, "is_anomaly": flags_r
            }), use_container_width=True)
    else:
        st.info("Select a channel and click **Predict**.")

# ── analysis tab ──────────────────────────────────────────────────────────────

with tab_analysis:
    st.subheader("Model evaluation")
    eval_data = load_eval_metrics()
    if eval_data:
        champion = eval_data.get("champion", "—")
        st.caption(f"Champion: **{champion}** · {eval_data.get('timestamp', '—')}")
        for key, label in [("isolation_forest", "Isolation Forest"), ("lstm", "LSTM Autoencoder")]:
            m = eval_data.get(key, {})
            if not m:
                continue
            is_champ = champion == key or champion == key.replace("_", " ")
            with st.expander(f"[champion] {label}" if is_champ else label, expanded=is_champ):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("F1", f"{m.get('f1', 0):.4f}")
                c2.metric("Precision", f"{m.get('precision', 0):.4f}")
                c3.metric("Recall", f"{m.get('recall', 0):.4f}")
                c4.metric("AUROC", f"{m.get('auroc', 0):.4f}")
                c5.metric("AUPRC", f"{m.get('auprc', 0):.4f}")
    else:
        st.info("No evaluation results found.")

    st.divider()
    st.subheader("Drift monitoring")
    flag = load_drift_flag()
    if flag:
        frac = flag["drift_fraction"] * 100
        n_drifted = flag["drifted_features"]
        total = flag["total_features"]
        if flag["retrain_required"]:
            st.warning(f"{n_drifted}/{total} features drifted ({frac:.1f}%) — retraining suggested")
        else:
            st.success(f"{n_drifted}/{total} features drifted ({frac:.1f}%) — no retraining needed")
        st.caption(
            f"Last checked: {flag.get('timestamp', '—')} · "
            f"Trigger threshold: >{flag['drift_fraction_threshold'] * 100:.0f}% of features"
        )
        st.caption(
            "Note: drift is measured between the training distribution and the synthetic test data. "
            "Because synthetic channels are generated independently via AR(1), the distributions will "
            "always differ slightly — a high drift fraction here is expected and does not reflect "
            "real sensor degradation."
        )

        per_feature = flag.get("per_feature", {})
        if per_feature:
            rows = [
                {"Feature": feat, "KS statistic": round(v["statistic"], 4),
                 "p-value": round(v["p_value"], 4), "Drifted": "Yes" if v["drifted"] else "No"}
                for feat, v in per_feature.items()
            ]
            drift_df = pd.DataFrame(rows).sort_values("KS statistic", ascending=False)
            left, right = st.columns(2)
            with left:
                st.markdown("**Per-feature KS results**")
                st.dataframe(drift_df, use_container_width=True, hide_index=True)
            with right:
                drifted_only = drift_df[drift_df["Drifted"] == "Yes"]
                if not drifted_only.empty:
                    fig_bar = go.Figure(go.Bar(
                        x=drifted_only["Feature"], y=drifted_only["KS statistic"],
                        marker_color=ANOMALY_COLOR,
                    ))
                    fig_bar.update_layout(xaxis_title="Feature", yaxis_title="KS statistic",
                                          margin=dict(l=20, r=20, t=20, b=40), height=300)
                    st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No drift report found.")

# ── data source tab ───────────────────────────────────────────────────────────

with tab_datasource:
    st.subheader("Dataset: NASA SMAP Telemetry")
    st.markdown("""
**SMAP** (Soil Moisture Active Passive) is a NASA satellite that monitors Earth's soil moisture.
Its onboard telemetry system continuously records sensor readings across multiple channels,
each representing a different subsystem or instrument group.

The dataset was published by Hundman et al. (2018) alongside the
[telemanom](https://github.com/khundman/telemanom) anomaly detection framework.
It contains **82 labeled channels** for the SMAP spacecraft, with ground-truth anomaly
intervals manually annotated by NASA engineers.

---

**Structure**

| Field | Description |
|---|---|
| Channel ID | Unique identifier per telemetry channel (e.g. `P-1`, `S-1`) |
| Features | 25 input features per timestep (raw sensor values) |
| Train split | Normal operating data only |
| Test split | Mix of normal and anomalous periods |
| Labels | `[start, end]` index pairs marking anomalous intervals |

---

**Synthetic data note**

The original `.npy` channel files are no longer publicly accessible.
This deployment uses **synthetic data** generated to match the real dataset's structure:

- Channel IDs and anomaly label locations are real (from `labeled_anomalies.csv`)
- Feature values are **AR(1) autoregressive** (φ=0.85, σ=0.3)
- Anomalies injected at labeled locations by shifting ~⅓ of features by ±3σ

---

**Pipeline**

```
labeled_anomalies.csv → ingestion → features → train (IF + LSTM) → evaluate → monitor → serve
```

**Reference:** Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*, KDD 2018.
    """)

# ── architecture tab ──────────────────────────────────────────────────────────

_NODE = (
    "flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;"
    "border-radius:8px;padding:10px 14px;"
    "text-align:center;font-size:0.78rem;line-height:1.35;border:1px solid rgba(0,0,0,0.08);"
)
_ARROW = "font-size:1.3rem;color:#94a3b8;padding:0 2px;align-self:center;flex-shrink:0;"
_ROW   = "display:flex;align-items:stretch;gap:4px;flex-wrap:nowrap;margin:8px 0;width:100%;"
_PHASE = (
    "border:1px dashed #94a3b8;border-radius:10px;padding:14px 16px 16px;"
    "margin-bottom:14px;"
)
_LABEL = "font-size:0.7rem;color:#94a3b8;letter-spacing:0.05em;margin-bottom:10px;"

# colour palette — background / border
_C = {
    "src":   ("background:#dbeafe;border-color:#93c5fd", "#1d4ed8"),   # blue   — data source
    "proc":  ("background:#fef9c3;border-color:#fde047", "#854d0e"),   # yellow — processing
    "model": ("background:#ede9fe;border-color:#c4b5fd", "#5b21b6"),   # purple — model
    "eval":  ("background:#fed7aa;border-color:#fdba74", "#9a3412"),   # orange — evaluation
    "out":   ("background:#d1fae5;border-color:#6ee7b7", "#065f46"),   # green  — output
    "user":  ("background:#fce7f3;border-color:#f9a8d4", "#9d174d"),   # pink   — user/query
}

def _node(label: str, sub: str, kind: str) -> str:
    bg, fg = _C[kind]
    return (
        f"<div style='{_NODE}{bg}'>"
        f"<b style='color:{fg}'>{label}</b>"
        f"<span style='color:#64748b;font-size:0.72rem;margin-top:3px'>{sub}</span>"
        f"</div>"
    )

def _arr() -> str:
    return f"<span style='{_ARROW}'>→</span>"

with tab_arch:
    st.subheader("System Architecture")
    st.markdown(
        "SensorFlow is split into two phases: an offline training pipeline that builds and "
        "evaluates models, and an online inference pipeline that scores telemetry channels in real time."
    )

    # ── diagram ───────────────────────────────────────────────────────────────

    offline_html = (
        f"<div style='{_PHASE}'>"
        f"<div style='{_LABEL}'>OFFLINE — ingestion &amp; training</div>"
        f"<div style='{_ROW}'>"
        + _node("NASA SMAP", "S3 / synthetic fallback", "src")
        + _arr()
        + _node("Ingestion", "per-channel .npy arrays", "proc")
        + _arr()
        + _node("Feature engineering", "lag features · 64-step windows", "proc")
        + _arr()
        + _node("Training", "Isolation Forest · LSTM Autoencoder", "model")
        + _arr()
        + _node("Evaluate &amp; select", "F1 · AUROC · champion", "eval")
        + "</div></div>"
    )

    online_html = (
        f"<div style='{_PHASE}'>"
        f"<div style='{_LABEL}'>ONLINE — prediction</div>"
        f"<div style='{_ROW}'>"
        + _node("Channel selector", "82 SMAP channels", "user")
        + _arr()
        + _node("AR(1) generator", "φ=0.85 · σ=0.3 · anomaly injection", "src")
        + _arr()
        + _node("StandardScaler", "fitted on training split", "proc")
        + _arr()
        + _node("Model inference", "IF score · LSTM recon. error", "model")
        + _arr()
        + _node("Anomaly scores", "threshold · score plot · signal plot", "out")
        + "</div></div>"
    )

    st.markdown(offline_html + online_html, unsafe_allow_html=True)

    st.divider()

    # ── component cards (full-width, stacked) ─────────────────────────────────

    st.markdown("##### Components")

    for title, body in [
        (
            "Data ingestion",
            "Raw telemetry is sourced from the NASA SMAP dataset (Hundman et al., 2018). "
            "Each channel is a separate time series with 25 sensor features. "
            "When the S3 bucket is unreachable, an AR(1) synthetic process (φ=0.85, σ=0.3) "
            "generates structurally equivalent data with anomalies injected at the real label "
            "locations by shifting ~⅓ of features by ±3σ.",
        ),
        (
            "Feature engineering",
            "Two feature representations are built from the raw arrays. "
            "For Isolation Forest: each timestep is expanded with 5 lag columns per feature, "
            "giving a flat vector of 150 dimensions. "
            "For LSTM: a sliding window of 64 timesteps produces 3-D tensors of shape "
            "(windows, 64, 25). Both representations are normalised with a StandardScaler "
            "fitted on the training split only.",
        ),
        (
            "Isolation Forest",
            "An ensemble of 100 isolation trees is trained on the flat lag-feature matrix. "
            "Each tree randomly selects a feature and a split value; points that are isolated "
            "quickly (short average path length) are scored as anomalous. "
            "The anomaly score is the negated decision function so that higher always means "
            "more anomalous, consistent with the LSTM scoring convention.",
        ),
        (
            "LSTM Autoencoder",
            "A 2-layer LSTM encoder compresses each 64-step window into a hidden state; "
            "a symmetric 2-layer LSTM decoder reconstructs the original sequence. "
            "The model is trained to minimise mean squared reconstruction error on normal "
            "windows only. Anomaly score = mean squared error over the window — "
            "high error means the model could not reconstruct the local temporal pattern. "
            "The detection threshold is the 95th percentile of training reconstruction errors.",
        ),
        (
            "Champion selection",
            "After training, both models are evaluated on the held-out test split using "
            "F1, precision, recall, AUROC, and AUPRC. The model with the higher F1 is "
            "designated champion and loaded by default in the Predict tab. "
            "Evaluation results are persisted to reports/eval_metrics.json and displayed "
            "in the Evaluate tab.",
        ),
        (
            "Drift monitoring",
            "A Kolmogorov-Smirnov test is run per feature between the training distribution "
            "and a reference sample of the test data. If more than 50% of features drift "
            "significantly (p < 0.05), a retraining flag is raised. "
            "Because synthetic train and test channels are generated independently, "
            "drift is expected to be high in this deployment and does not indicate "
            "real sensor degradation.",
        ),
        (
            "Deployment",
            "The app runs on Streamlit Community Cloud (free tier). "
            "Trained model weights are committed directly to the GitHub repository "
            "(IF ~2.5 MB, LSTM ~455 KB) so no external model store is needed at startup. "
            "The FastAPI serving layer and MLflow experiment tracker run locally; "
            "this Space loads models directly from disk with no API dependency.",
        ),
    ]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(body)
