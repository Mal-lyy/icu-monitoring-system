# 🏥 Real-Time ICU Patient Vitals Monitoring System

> A Big Data pipeline for clinical intelligence and mortality prediction  
> BIA 678-WS | Big Data Technologies | Stevens Institute of Technology | May 2026

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-KRaft-black?logo=apachekafka)](https://kafka.apache.org)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.1-orange?logo=apachespark)](https://spark.apache.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-NoSQL-green?logo=mongodb)](https://mongodb.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-AUC%200.9467-red)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This system streams 330 million ICU vital sign readings from the 
MIMIC-III Clinical Database through a Lambda Architecture pipeline, 
scoring every reading with five parallel layers of clinical rules 
and machine learning in real time. Results are displayed on a live 
Streamlit dashboard that refreshes every 5 seconds.

The XGBoost model achieves AUC-ROC **0.9467** — a **15.4 percentage 
point improvement** over the NHS gold-standard NEWS2 scoring system 
on the same independent test set of 12,038 patients.

---

## Architecture

![Pipeline Architecture](assets/architecture.png)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Message Broker | Apache Kafka (KRaft mode) |
| Stream Processor | Apache Spark 3.5.1 (PySpark) |
| Database | MongoDB |
| Dashboard | Streamlit + Plotly |
| ML Framework | XGBoost + PyTorch (LSTM) |
| Explainability | SHAP (TreeExplainer) |
| Containerisation | Docker Desktop |
| Data Source | MIMIC-III (PhysioNet) |

---

## Five Scoring Layers

| Layer | Method | Trigger |
|-------|--------|---------|
| 1 | NEWS2 (RCP 2017) | Score ≥ 7 or red score = CRITICAL |
| 2 | qSOFA (Singer JAMA 2016) | Score ≥ 2 = CRITICAL |
| 3 | Shock Index (HR/SBP) | > 1.0 = WARNING |
| 4 | XGBoost (53 features) | ≥ 70% mortality = WARNING |
| 5 | LSTM 48h (24 readings) | ≥ 50% = WARNING |

---

## Model Results

| Model | AUC-ROC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| Charlson CCI | 0.6444 | 0.6029 | 0.1549 | 0.2465 |
| qSOFA | 0.6716 | 0.5166 | 0.2458 | 0.3331 |
| NEWS2 (NHS gold standard) | 0.7930 | 0.9206 | 0.1482 | 0.2553 |
| Shock Index | 0.7936 | 0.8350 | 0.1753 | 0.2898 |
| Logistic Regression | 0.9055 | 0.7995 | 0.4054 | 0.5380 |
| **XGBoost** | **0.9467** | 0.7726 | 0.6014 | 0.6763 |
| LSTM 48h | 0.9469 | 0.3300 | 0.7500 | 0.4600 |

---

## Dashboard

![Dashboard Overview](assets/dashboard_overview.png)
![Patient Detail](assets/patient_detail.png)

---

## Project Structure

icu-monitoring-system/
├── simulator/
│   └── producer.py              # Kafka producer — streams MIMIC-III
├── spark/
│   └── stream_processor.py      # 5-layer real-time scoring engine
├── dashboard/
│   └── app.py                   # Streamlit web dashboard
├── sepsis_detector.py           # Batch: Sepsis-3 lab analysis
├── charlson_scorer.py           # Batch: Charlson CCI scoring
├── demographics_enricher.py     # Batch: patient enrichment
├── feature_engineering.py       # Builds 53-feature matrix
├── prepare_sequences.py         # LSTM sequence preparation
├── train_xgboost.py             # XGBoost training pipeline
├── train_lstm.py                # LSTM training pipeline
├── compare_models.py            # Model evaluation and comparison
├── test_icu_project.py          # 62 automated tests (100% pass)
├── docker-compose.yml           # Kafka + MongoDB containers
├── requirements.txt             # Python dependencies
└── start.sh                     # One-command pipeline launch

---

## Data Access

This project uses the **MIMIC-III Clinical Database** which requires 
credentialed access through PhysioNet due to patient privacy regulations.

To get access:
1. Complete the [CITI Data or Specimens Only Research](https://about.citiprogram.org) training
2. Apply at [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
3. Sign the data use agreement
4. Place CSV files in the project root

> ⚠️ MIMIC-III data files are not included in this repository as 
> per the PhysioNet data use agreement.

---

## Pre-trained Models

Download the pre-trained model bundle from 
[Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE) and place in `models/`:
- `xgboost_mortality.pkl`
- `lstm_48h.pkl`
- `lstm_hospital.pkl`

---

## Setup and Installation

### Prerequisites
- Python 3.11+
- Docker Desktop
- Java 17 (for Spark)
- 16GB RAM recommended

### Install dependencies
```bash
pip install -r requirements.txt
brew install libomp  # Mac only, for XGBoost
```

### Run the pipeline
```bash
# Start Docker containers
docker-compose up -d

# Run batch layer (once)
python sepsis_detector.py
python charlson_scorer.py
python demographics_enricher.py

# Start streaming pipeline
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
    spark/stream_processor.py

# Start producer
python simulator/producer.py

# Launch dashboard
streamlit run dashboard/app.py
```

Or simply:
```bash
./start.sh
```

Open `http://localhost:8501` in your browser.

---

## Testing

```bash
python -m pytest test_icu_project.py -v
```

62 tests, 100% pass rate across 10 sections: data integrity, 
clinical scoring boundaries, MongoDB schema, ML model validity, 
and end-to-end integration.

---

**Course:** BIA 678-WS — Big Data Technologies  
**Institution:** Stevens Institute of Technology  
**Semester:** Spring 2026

---

## References

Key clinical references used:
- Johnson et al. MIMIC-III (Scientific Data, 2016)
- Royal College of Physicians. NEWS2 (2017)
- Singer et al. Sepsis-3 (JAMA, 2016)
- Harutyunyan et al. MIMIC Benchmarks (Scientific Data, 2019)
- Deng et al. LSTM ICU Mortality (Frontiers in Medicine, 2022)
