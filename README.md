# Agro-IoT Yield Optimization

> **Precision agriculture platform for a leading LATAM agro-industrial operation** — IoT sensor ingestion, automated ETL pipelines, and ML-based crop yield prediction reducing data reprocessing by 25% and cutting input costs by 12%.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Apache%20Airflow-2.8-017CEE?logo=apacheairflow&logoColor=white" />
  <img src="https://img.shields.io/badge/IoT%20Sensors-MQTT-660066" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 📌 Project Overview

This repository contains the data engineering and machine learning platform built for a large-scale agro-industrial operation managing thousands of hectares across multiple growing regions in LATAM. The system ingests real-time sensor telemetry (soil moisture, temperature, humidity, NDVI proxies), transforms it through automated Airflow pipelines, and generates per-lot yield predictions to guide precision irrigation and fertilization decisions.

> 🌐 **Full project details and case study:** [bravoaidatastudio.com](https://bravoaidatastudio.com/portfolio/)

**The business challenge:** Manual field monitoring across thousands of hectares was reactive, labor-intensive, and inconsistent. Agronomic decisions were based on periodic physical inspections rather than continuous data, causing suboptimal input application and yield variability between lots.

---

## 📊 Business Impact

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Data reprocessing hours / week | ~18 hrs manual | ~2 hrs automated | **−25% rework** |
| Water & fertilizer cost per hectare | Baseline | Optimized per-lot | **−12% input cost** |
| Monitoring latency | Days (manual inspection) | Minutes (real-time telemetry) | **Continuous** |
| Yield prediction error (MAE) | No model | ~8.2 qt/ha | **Operational** |
| Lots with automated alerts | 0 | All active lots | **100% coverage** |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FIELD LAYER (IoT)                             │
│  Soil Moisture  │  Temperature  │  Humidity  │  NDVI Proxy  │  Rain  │
│            Sensors transmit via MQTT every 15 minutes                │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    INGESTION & ORCHESTRATION                          │
│            Apache Airflow DAGs — incremental sensor ingestion        │
│            Data lake landing zone (Parquet / S3-compatible)          │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER (Python)                          │
│    Cleaning · Normalization · Outlier detection · Feature assembly   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ML PREDICTION (scikit-learn)                       │
│    Random Forest Regressor — per-lot yield prediction (qt/ha)        │
│    Irrigation & fertilization recommendation engine                  │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│              OPERATIONAL LAYER  (Dashboards / Alerts)                │
│    Agronomic dashboard  │  Low-yield alerts  │  Input recommendations│
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Sensor Variables & Features

| Sensor / Variable | Unit | Agronomic Relevance |
|---|---|---|
| Soil moisture (0–30 cm) | % vol. | Direct irrigation trigger |
| Soil temperature (0–10 cm) | °C | Germination & root activity |
| Air temperature | °C | Evapotranspiration proxy |
| Relative humidity | % | Disease pressure indicator |
| Rainfall | mm/15 min | Irrigation offset |
| Solar radiation proxy | W/m² | Photosynthesis potential |
| NDVI proxy (optical sensor) | 0–1 | Canopy health indicator |
| pH (periodic) | pH units | Nutrient availability |

### Engineered Features for ML

- **Rolling aggregates** — 24h / 72h / 7d mean, min, max for each sensor
- **Stress indicators** — consecutive hours below moisture threshold, heat stress days
- **Growth stage** — days after sowing (DAS) encoded as continuous + categorical bins
- **Lot attributes** — area (ha), soil type, crop variety, elevation
- **Cumulative GDD** — Growing Degree Days accumulated since sowing

---

## 📂 Repository Structure

```
Agro-IoT-Yield-Optimization/
├── airflow_dags/
│   └── sensor_ingestion_dag.py       # Airflow DAG: 15-min sensor → data lake
├── notebooks/
│   └── 01_EDA_sensor_yield.ipynb     # Exploratory analysis: sensors × yield
├── src/
│   ├── simulation/
│   │   └── sensor_simulator.py       # Realistic sensor telemetry generator
│   ├── processing/
│   │   └── sensor_processor.py       # Cleaning, normalization, feature engineering
│   └── models/
│       └── train_model.py            # Random Forest yield predictor
├── pipeline_showcase.py              # End-to-end walkthrough (run this first)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone & install**
```bash
git clone https://github.com/aDavidBravo/Agro-IoT-Yield-Optimization.git
cd Agro-IoT-Yield-Optimization
pip install -r requirements.txt
```

**2. Generate sensor telemetry data**
```bash
python src/simulation/sensor_simulator.py
# → outputs data/sensor_readings.parquet  (~500k readings across 3 growing seasons)
```

**3. Run the EDA notebook**
```bash
jupyter notebook notebooks/01_EDA_sensor_yield.ipynb
```

**4. Run the full pipeline showcase**
```bash
python pipeline_showcase.py
# → trains model, generates yield predictions, saves explainability dashboard
```

---

## 📈 Key Findings from EDA

- Soil moisture below **28% vol.** for >48 consecutive hours correlates with **−18% yield** vs irrigated lots
- **Solar radiation** and **GDD accumulation** are the two strongest individual predictors of final yield (combined importance ~34%)
- Lots at **higher elevation** (>800 m) show lower yield variance but are more sensitive to temperature drops
- **pH outside 6.0–7.2** reduces predicted yield by up to 22% regardless of irrigation adequacy
- Night temperature variance >8°C during grain fill stage is a reliable stress indicator

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | Apache Airflow 2.8 |
| Sensor Simulation | Python (NumPy, pandas) |
| Data Storage | Parquet (S3-compatible) |
| Processing | pandas, scikit-learn pipelines |
| ML Model | Random Forest Regressor |
| Explainability | SHAP, feature importance |
| Visualization | Matplotlib, Seaborn |

---

## 👤 Author

**David Bravo** — Data Scientist & AI Solutions Architect

> 🌐 [bravoaidatastudio.com](https://bravoaidatastudio.com) | 📁 [Portfolio](https://bravoaidatastudio.com/portfolio/)

*This project was developed under confidentiality agreement for a leading LATAM agro-industrial group. All data in this repository is synthetic and statistically calibrated to reflect real agronomic patterns.*

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
