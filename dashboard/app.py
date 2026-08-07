"""SensorFlow Streamlit dashboard.

Layout
------
Sidebar   : API URL, channel selector, Predict button
Main area :
  Row 1 — stat tiles: model type, alias, uptime, total requests, anomaly rate
  Row 2 — anomaly score plot (full width, interactive)
  Row 3 — drift summary table  |  raw per-feature KS results
  Footer  — link to Evidently HTML report and OpenAPI docs
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SensorFlow",
    page_icon="🛰️",
    layout="wide",
)

# ── constants ─────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
DEFAULT_API = "http://localhost:8000"
ANOMALY_COLOR = "#ef4444"   # red
NORMAL_COLOR = "#3b82f6"    # blue
THRESHOLD_COLOR = "#f59e0b" # amber


# ── helpers ───────────────────────────────────────────────────────────────────


@st.cache_data(ttl=30)
def _get_health(api_url: str) -> dict:
    try:
        r = httpx.get(f"{api_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}


@st.cache_data(ttl=30)
def _get_api_metrics(api_url: str) -> dict:
    try:
        r = httpx.get(f"{api_url}/metrics", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


@st.cache_data
def _load_test_channels() -> dict[str, np.ndarray]:
    """Load test parquet and return {chan_id: readings_array}."""
    test_path = DATA_DIR / "test.parquet"
    if not test_path.exists():
        return {}
    df = pd.read_parquet(test_path)
    feat_cols = sorted(c for c in df.columns if c.startswith("f_"))
    channels: dict[str, np.ndarray] = {}
    for chan_id, grp in df.groupby("channel_id", sort=False):
        channels[str(chan_id)] = grp[feat_cols].values.astype(float)
    return channels


@st.cache_data
def _load_drift_flag() -> dict | None:
    flag_path = REPORTS_DIR / "retrain_flag.json"
    if not flag_path.exists():
        return None
    return json.loads(flag_path.read_text(encoding="utf-8"))


@st.cache_data(ttl=60)
def _load_eval_metrics() -> dict | None:
    path = REPORTS_DIR / "eval_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _predict(api_url: str, channel_id: str, readings: np.ndarray) -> dict | None:
    try:
        r = httpx.post(
            f"{api_url}/predict",
            json={"channel_id": channel_id, "readings": readings.tolist()},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        st.error(f"API error {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:
        st.error(f"Could not reach API: {exc}")
    return None


def _signal_plot(readings: np.ndarray, predictions: list[dict], channel_id: str) -> go.Figure:
    """Raw sensor signal with anomalous timesteps shaded."""
    anom_set = {p["timestep"] for p in predictions if p["is_anomaly"]}
    n_timesteps, n_features = readings.shape
    timesteps = list(range(n_timesteps))

    fig = go.Figure()
    for i in range(n_features):
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=readings[:, i].tolist(),
            mode="lines",
            line=dict(width=1),
            opacity=0.5,
            name=f"f_{i:02d}",
            showlegend=False,
        ))

    # Shade anomalous regions as vertical bands.
    if anom_set:
        sorted_anom = sorted(anom_set)
        # Group consecutive timesteps into contiguous spans.
        spans, start = [], sorted_anom[0]
        prev = sorted_anom[0]
        for t in sorted_anom[1:]:
            if t > prev + 1:
                spans.append((start, prev))
                start = t
            prev = t
        spans.append((start, prev))

        y_min = float(readings.min())
        y_max = float(readings.max())
        for s, e in spans:
            fig.add_shape(type="rect", x0=s, x1=e, y0=y_min, y1=y_max,
                          fillcolor=ANOMALY_COLOR, opacity=0.15, line_width=0, layer="below")

    fig.update_layout(
        title=f"Raw signal — channel {channel_id} ({n_features} features)",
        xaxis_title="Timestep",
        yaxis_title="Value",
        margin=dict(l=40, r=20, t=60, b=40),
        height=300,
    )
    return fig


def _score_plot(
    predictions: list[dict],
    channel_id: str,
    threshold: float | None = None,
) -> go.Figure:
    timesteps = [p["timestep"] for p in predictions]
    scores = [p["score"] for p in predictions]
    flags = [p["is_anomaly"] for p in predictions]

    normal_x = [t for t, f in zip(timesteps, flags) if not f]
    normal_y = [s for s, f in zip(scores, flags) if not f]
    anom_x = [t for t, f in zip(timesteps, flags) if f]
    anom_y = [s for s, f in zip(scores, flags) if f]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal_x, y=normal_y, mode="markers",
        marker=dict(color=NORMAL_COLOR, size=3, opacity=0.6),
        name="Normal",
    ))
    fig.add_trace(go.Scatter(
        x=anom_x, y=anom_y, mode="markers",
        marker=dict(color=ANOMALY_COLOR, size=6, symbol="x"),
        name="Anomaly",
    ))
    y_max = max(scores) if scores else 1.0
    if threshold is not None:
        y_max = max(y_max, threshold)
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"threshold={threshold:.4f}",
            annotation_position="top right",
        )
    fig.update_layout(
        title=f"Anomaly scores — channel {channel_id}",
        xaxis_title="Timestep",
        yaxis_title="Anomaly score",
        yaxis=dict(range=[0, y_max * 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    return fig


def _drift_table(flag: dict) -> pd.DataFrame:
    rows = []
    for feat, result in flag["per_feature"].items():
        rows.append({
            "Feature": feat,
            "KS statistic": round(result["statistic"], 4),
            "p-value": round(result["p_value"], 4),
            "Drifted": "⚠️ Yes" if result["drifted"] else "✅ No",
        })
    return pd.DataFrame(rows).sort_values("KS statistic", ascending=False)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("SensorFlow")
    st.caption("NASA SMAP anomaly detection")
    st.divider()
    api_url = st.text_input("API URL", value=DEFAULT_API)
    st.divider()
    health = _get_health(api_url)
    api_metrics = _get_api_metrics(api_url)
    st.metric("Status", health.get("status", "—").upper())
    st.metric("Model", health.get("model_type") or "—")
    st.metric("Alias", health.get("model_alias") or "—")
    st.metric("Requests", api_metrics.get("total_requests", "—"))
    st.metric(
        "Anomaly rate",
        f"{api_metrics.get('anomaly_rate', 0) * 100:.1f}%" if api_metrics else "—",
    )
    st.caption("Polling /health every 30 s")


channels = _load_test_channels()

# ── tabs ──────────────────────────────────────────────────────────────────────

tab_predict, tab_analysis, tab_datasource = st.tabs(["Predict", "Analysis", "Data Source"])


# ── predict tab ───────────────────────────────────────────────────────────────

with tab_predict:
    ctrl_col, _ = st.columns([1, 2])
    with ctrl_col:
        if channels:
            channel_id = st.selectbox("Channel", sorted(channels.keys()))
        else:
            st.warning("No test data found. Run ingestion first.")
            channel_id = None
        predict_btn = st.button("Predict", disabled=channel_id is None, use_container_width=True)

    if predict_btn and channel_id:
        with st.spinner(f"Scoring channel {channel_id} …"):
            result = _predict(api_url, channel_id, channels[channel_id])
            if result:
                st.session_state["last_result"] = result
                st.session_state["last_channel"] = channel_id

    result = st.session_state.get("last_result")
    last_channel = st.session_state.get("last_channel", channel_id)

    if result:
        preds = result["predictions"]
        n_anom = sum(p["is_anomaly"] for p in preds)
        st.subheader(f"Channel {last_channel} — {n_anom} anomalies in {len(preds)} scored windows")

        threshold = result.get("threshold")
        st.plotly_chart(_score_plot(preds, last_channel, threshold), use_container_width=True)

        raw = channels.get(last_channel)
        if raw is not None:
            st.plotly_chart(_signal_plot(raw, preds, last_channel), use_container_width=True)

        with st.expander("Raw prediction data"):
            st.dataframe(pd.DataFrame(preds), use_container_width=True)
    else:
        st.info("Select a channel and click **Predict** to see anomaly scores.")


# ── analysis tab ──────────────────────────────────────────────────────────────

with tab_analysis:
    st.subheader("Model evaluation")
    eval_data = _load_eval_metrics()

    if eval_data:
        champion = eval_data.get("champion", "—")
        st.caption(f"Champion: **{champion}** · Last evaluated: {eval_data.get('timestamp', '—')}")

        for model_key, label in [("isolation_forest", "Isolation Forest"), ("lstm", "LSTM Autoencoder")]:
            m = eval_data.get(model_key, {})
            if not m:
                continue
            is_champ = champion == model_key.replace("_", " ") or champion == model_key
            with st.expander(f"{'🏆 ' if is_champ else ''}{label}", expanded=is_champ):
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("F1", f"{m.get('f1', 0):.4f}")
                c2.metric("Precision", f"{m.get('precision', 0):.4f}")
                c3.metric("Recall", f"{m.get('recall', 0):.4f}")
                c4.metric("AUROC", f"{m.get('auroc', 0):.4f}")
                c5.metric("AUPRC", f"{m.get('auprc', 0):.4f}")
    else:
        st.info("No evaluation results found. Run evaluate.py first.")

    st.divider()
    st.subheader("Drift monitoring")
    flag = _load_drift_flag()

    if flag:
        retrain = flag["retrain_required"]
        frac = flag["drift_fraction"] * 100
        n_drifted = flag["drifted_features"]
        total = flag["total_features"]

        banner_col, meta_col = st.columns([2, 1])
        with banner_col:
            if retrain:
                st.error(f"Retraining required — {n_drifted}/{total} features drifted ({frac:.1f}%)")
            else:
                st.success(f"No retraining needed — {n_drifted}/{total} features drifted ({frac:.1f}%)")
        with meta_col:
            st.caption(f"Last checked: {flag.get('timestamp', '—')}")
            st.caption(f"Threshold: {flag['drift_fraction_threshold'] * 100:.0f}% of features")

        left, right = st.columns(2)
        drift_df = _drift_table(flag)
        with left:
            st.markdown("**Per-feature KS results** (sorted by statistic)")
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
        with right:
            drifted_only = drift_df[drift_df["Drifted"] == "⚠️ Yes"]
            st.markdown(f"**{len(drifted_only)} drifted features**")
            if not drifted_only.empty:
                fig_bar = go.Figure(go.Bar(
                    x=drifted_only["Feature"],
                    y=drifted_only["KS statistic"],
                    marker_color=ANOMALY_COLOR,
                ))
                fig_bar.update_layout(
                    xaxis_title="Feature", yaxis_title="KS statistic",
                    margin=dict(l=20, r=20, t=20, b=40), height=300,
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No drifted features.")
    else:
        st.info("No drift report found. Run monitor.py first.")

    st.divider()
    fc1, fc2 = st.columns(2)
    fc1.markdown(f"[Evidently drift report]({api_url.rstrip('/')}/reports/drift_report.html)")
    fc2.markdown(f"[API docs]({api_url.rstrip('/')}/docs)")


# ── data source tab ───────────────────────────────────────────────────────────

with tab_datasource:
    st.subheader("Dataset: NASA SMAP Telemetry")
    st.markdown("""
**SMAP** (Soil Moisture Active Passive) is a NASA satellite that monitors Earth's soil moisture.
Its onboard telemetry system continuously records sensor readings across multiple channels,
each representing a different subsystem or instrument group.

The dataset used here was originally published by Hundman et al. (2018) alongside the
[telemanom](https://github.com/khundman/telemanom) anomaly detection framework.
It contains **82 labeled channels** for the SMAP spacecraft, with ground-truth anomaly
intervals manually annotated by NASA engineers.

---

**Structure**

| Field | Description |
|---|---|
| Channel ID | Unique identifier per telemetry channel (e.g. `P-1`, `S-1`) |
| Features | 25 input features per timestep (raw sensor values) |
| Train split | Normal operating data only — no labeled anomalies |
| Test split | Mix of normal and anomalous periods |
| Labels | `[start, end]` index pairs marking anomalous intervals |

---

**Synthetic fallback**

The original `.npy` channel files are no longer publicly accessible (S3 bucket taken down).
This deployment uses **synthetic data** generated to match the real dataset's structure:

- Channel IDs and anomaly label locations are real (from the original `labeled_anomalies.csv`)
- Feature values are generated as an **AR(1) autoregressive process** (φ=0.85, σ=0.3)
  to produce realistic autocorrelated telemetry-like signals
- Anomalies are injected at the labeled locations by shifting ~⅓ of features by ±3σ

The models train and evaluate on this synthetic data, which preserves the structural
and statistical properties of real SMAP telemetry while keeping the full pipeline runnable.

---

**Pipeline**

```
labeled_anomalies.csv  →  ingestion.py   →  train/val/test .parquet
                       →  features.py    →  sliding windows + lag features
                       →  train.py       →  Isolation Forest + LSTM Autoencoder
                       →  evaluate.py    →  F1 / AUROC / drift report
                       →  monitor.py     →  per-feature KS drift detection
                       →  serve.py       →  FastAPI /predict
```

**Reference:** Hundman et al., *Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding*, KDD 2018.
    """)

