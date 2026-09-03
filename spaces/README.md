---
title: SensorFlow
emoji: 🛰️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# 📡 SensorFlow

Anomaly detection for NASA SMAP satellite telemetry. Two models are trained on 82 labeled satellite sensor channels and compared automatically — the better one is selected and used for predictions.

## What it does

- Scores any SMAP telemetry channel for anomalies using an LSTM Autoencoder or Isolation Forest
- Highlights time windows where sensor readings deviate from learned normal behaviour
- Evaluates both models against NASA ground-truth labels and promotes the champion by F1
- Monitors feature drift between training and live data and flags when retraining is recommended

## Tabs

| Tab | Description |
|---|---|
| **Predict** | Pick any of the 82 SMAP channels and run anomaly detection |
| **Evaluate** | Compare both models on F1, precision, recall, AUROC, and AUPRC |
| **Data Sources** | Dataset background, structure, and synthetic data note |
| **Architecture** | System diagram and component descriptions |

## Sidebar

Three collapsible sections explain the tool in plain language for non-technical readers. A separate section shows the current best-performing model and its anomaly threshold.

## Stack

- **Models**: Isolation Forest · LSTM Autoencoder (2-layer, 64-step windows)
- **Data**: Synthetic AR(1) channels (φ=0.85, σ=0.3) matching real SMAP structure; anomalies injected at NASA-labeled locations
- **Pipeline**: ingestion → features → train → evaluate → monitor
- **App**: Streamlit — models loaded from disk, no external API needed
