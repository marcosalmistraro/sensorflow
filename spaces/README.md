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

# SensorFlow

Anomaly detection for NASA SMAP satellite telemetry.

- **Models**: Isolation Forest + LSTM Autoencoder
- **Data**: Synthetic AR(1) channels matching real SMAP structure
- **Pipeline**: ingestion → features → train → evaluate → monitor
