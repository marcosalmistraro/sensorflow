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
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=THRESHOLD_COLOR,
            annotation_text=f"threshold={threshold:.4f}",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title=f"Anomaly scores — channel {channel_id}",
        xaxis_title="Timestep",
        yaxis_title="Anomaly score",
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
    st.title("🛰️ SensorFlow")
    st.caption("NASA SMAP anomaly detection")
    st.divider()

    api_url = st.text_input("API URL", value=DEFAULT_API)

    channels = _load_test_channels()
    if channels:
        channel_id = st.selectbox("Channel", sorted(channels.keys()))
    else:
        st.warning("No test data found. Run ingestion.py first.")
        channel_id = None

    predict_btn = st.button("🔍 Predict", disabled=channel_id is None, use_container_width=True)
    st.divider()
    st.caption("Polling /health every 30 s")


# ── stat tiles ────────────────────────────────────────────────────────────────

health = _get_health(api_url)
api_metrics = _get_api_metrics(api_url)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Status", health.get("status", "—").upper())
col2.metric("Model type", health.get("model_type") or "—")
col3.metric("Alias", health.get("model_alias") or "—")
col4.metric("Total requests", api_metrics.get("total_requests", "—"))
col5.metric(
    "Anomaly rate",
    f"{api_metrics.get('anomaly_rate', 0) * 100:.1f}%" if api_metrics else "—",
)

st.divider()

# ── prediction ────────────────────────────────────────────────────────────────

if predict_btn and channel_id:
    with st.spinner(f"Scoring channel {channel_id} …"):
        readings = channels[channel_id]
        result = _predict(api_url, channel_id, readings)
        if result:
            st.session_state["last_result"] = result
            st.session_state["last_channel"] = channel_id

result = st.session_state.get("last_result")
last_channel = st.session_state.get("last_channel", channel_id)

if result:
    preds = result["predictions"]
    n_anom = sum(p["is_anomaly"] for p in preds)
    st.subheader(f"Channel {last_channel} — {n_anom} anomalies in {len(preds)} scored windows")

    scores = [p["score"] for p in preds]
    threshold = float(np.percentile(scores, 95)) if scores else None
    st.plotly_chart(
        _score_plot(preds, last_channel, threshold),
        use_container_width=True,
    )

    with st.expander("Raw prediction data"):
        st.dataframe(pd.DataFrame(preds), use_container_width=True)
else:
    st.info("Select a channel and click **Predict** to see anomaly scores.")

st.divider()

# ── drift section ─────────────────────────────────────────────────────────────

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
            st.error(f"⚠️ Retraining required — {n_drifted}/{total} features drifted ({frac:.1f}%)")
        else:
            st.success(f"✅ No retraining needed — {n_drifted}/{total} features drifted ({frac:.1f}%)")
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
                xaxis_title="Feature",
                yaxis_title="KS statistic",
                margin=dict(l=20, r=20, t=20, b=40),
                height=300,
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No drifted features.")
else:
    st.info("No drift report found. Run monitor.py first.")

st.divider()

# ── footer ────────────────────────────────────────────────────────────────────

fc1, fc2 = st.columns(2)
fc1.markdown(f"[📄 Evidently drift report]({api_url.rstrip('/')}/reports/drift_report.html)")
fc2.markdown(f"[📖 API docs]({api_url.rstrip('/')}/docs)")
