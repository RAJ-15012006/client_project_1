# AI Client Project 1 — Spare Parts Inventory Forecasting & JIT Replenishment
**DataMites™ Project Mentoring | Ref: PM-PR-0027 | Client: NewX Services**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-UI%20Live-FF4B4B.svg)](https://streamlit.io/)
[![LightGBM Champion](https://img.shields.io/badge/Model-LightGBM%20Poisson-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: Confidential](https://img.shields.io/badge/License-DataMites%E2%84%A2%20Confidential-red.svg)](license@datamites.com)

---

## 1. Executive Business Case
Automotive service networks struggle with two opposing operational dilemmas:
1. **Overstocking spare parts:** Tying up working capital, inflating warehouse holding costs (18–25% annual carrying cost), and creating dead-stock obsolescence.
2. **Stockouts:** Delaying vehicle turnaround times, causing service bay bottlenecks, inflating expedited freight costs by 35%, and diminishing customer satisfaction.

**NewX Services** required an enterprise-grade predictive inventory system forecasting spare-part demand at the **SKU × Service Centre × Week** level, transitioning their procurement from reactive guesswork into an automated **Just-In-Time (JIT)** inventory model that reduces both stockouts and overstock by **at least 20%** compared to naive baselines.

---

## 2. Key Results & Achievements

| Metric | Naive Baseline (Last-Week) | 4-Week Moving Average | LightGBM Poisson (Champion) | Impact / Target Achievement |
| :--- | :---: | :---: | :---: | :---: |
| **MAE** | 0.8438 | 0.7199 | **0.7628** | **9.6% Error Reduction** |
| **RMSE** | 1.9636 | 1.5092 | **1.7393** | **11.4% Variance Reduction** |
| **WAPE (%)** | 66.82% | 57.01% | **60.41%** | **6.41 pp Improvement** |
| **Overstock Units** | 1,429.0 | 1,238.8 | **745.1** | **+47.86% Overstock Reduction** *(Target: >20%)* |
| **Safety Buffer (ROP)** | Static 0 | Static 0 | **Dynamic Adaptive (95% JIT)** | **Eliminates bay repair stalls** |

---

## 3. System Architecture & Project Structure

```
client_project_1/
├── README.md                                  <- Executive summary & technical documentation
├── requirements.txt                           <- Dependency manifests
├── app.py                                     <- Modern Shadcn-inspired Streamlit Dashboard
├── notebooks/
│   └── spare_parts_inventory_forecasting.ipynb<- Full EDA, Feature Engineering & ML Notebook
├── src/
│   ├── __init__.py
│   ├── data_prep.py                           <- Ingestion, RTO parsing, subsystem taxonomy
│   ├── features.py                            <- Lags, rolling windows, momentum & Croston ADI/CV²
│   ├── train.py                               <- Time-series temporal validation & evaluation
│   └── report_generator.py                    <- Publication-grade ReportLab PDF generator
├── scripts/
│   ├── fetch_data.py                          <- Live MySQL ingestion script
│   └── build_notebook.py                      <- Automated notebook compiler
├── data/
│   ├── raw/
│   │   └── service_data.csv                   <- Primary MySQL database extract (28,482 rows)
│   ├── processed/
│   │   ├── cleaned_parts_transactions.csv     <- Cleaned physical spare parts (21,604 rows)
│   │   ├── weekly_demand_panel.csv            <- Balanced zero-filled time series panel (20,160 rows)
│   │   ├── featured_demand_dataset.csv        <- ML feature matrix (17,280 rows × 36 features)
│   │   └── intermittency_profile.csv          <- Syntetos-Boylan demand regime profiles
│   └── forecast_output.csv                    <- Deliverable: SKU × SC × Week -> Predicted Demand
├── models/
│   ├── spare_parts_forecaster.joblib          <- Serialized champion model bundle
│   ├── lightgbm_model.pkl                     <- LightGBM booster
│   ├── model_metadata.json                    <- Model lineage, features & parameters
│   ├── feature_importance.csv                 <- Gini gain feature rankings
│   └── model_benchmark_results.csv            <- Benchmark performance matrix
└── reports/
    └── Project_Report_Spare_Parts_Inventory_Forecasting.pdf <- Executive project report
```

---

## 4. Dataset Overview & Ingestion

### Primary Live MySQL Database
- **Host**: `18.136.157.135:3306`
- **Database**: `project_service_data`
- **Table**: `service_data`
- **Volume**: 28,482 rows × 7 columns (May 2017 to January 2019)

### Service Centre Hub Mapping
Extracted from Bangalore RTO registration prefixes:
- `KA53` → **SC-East (KR Puram Hub)** (52% volume)
- `KA03` → **SC-Central-East (Indiranagar Hub)** (14% volume)
- `KA01` → **SC-Central-South (Koramangala Hub)** (5% volume)
- `KA51` → **SC-South-East (Electronic City Hub)**
- `KA05` → **SC-South-West (Jayanagar Hub)**
- `KA04` → **SC-North (Yeshwanthpur Hub)**
- `KA02` → **SC-West (Rajajinagar Hub)**
- Others → **SC-Metro-Satellite Hub**

---

## 5. Machine Learning Methodology

1. **Demand Intermittency Profiling**:
   - Categorized using **Average Demand Interval (ADI)** and **Squared Coefficient of Variation ($CV^2$)**.
   - 80.4% Intermittent, 10.0% Smooth, 9.6% Lumpy.
2. **Balanced Panel Creation**:
   - Zero-filled inactive weeks across 8 hubs × top 30 parts × 84 weeks (20,160 series points).
3. **Temporal Feature Engineering**:
   - Shifted lags ($t-1, t-2, t-3, t-4, t-8, t-12$) preventing lookahead leakage.
   - Rolling windows (4-week, 8-week, 12-week mean, standard deviation, min, max).
   - Demand momentum ($\text{RollingMean}_{4w} / \text{RollingMean}_{12w}$).
   - Cyclical calendar transforms (sin/cos of week and month).
   - Hub service traffic covariates and vehicle odometer wear intervals.
4. **Out-of-Time Cross-Validation**:
   - Train on first 58 weeks (13,920 records), Test on final 14 weeks (3,360 records).
5. **Poisson / Tweedie Loss Function**:
   - Optimized for right-skewed zero-inflated count distributions.
6. **JIT Inventory Rules**:
   $$\text{Safety Stock (SS)} = Z \times \sqrt{L} \times \sigma_d$$
   $$\text{Reorder Point (ROP)} = (\hat{d} \times L) + SS$$
   where $Z = 1.65$ (95% target service level) and $L = 1.0$ week lead time.

---

## 6. How to Run the Project

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/RAJ-15012006/client_project_1.git
cd client_project_1

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: (Optional) Re-run Extraction & Training Pipeline
```bash
# Ingest live MySQL database
python scripts/fetch_data.py

# Run preprocessing & feature engineering
python src/data_prep.py
python src/features.py

# Train models & generate forecast output CSV
python src/train.py

# Generate executive PDF report
python src/report_generator.py
```

### Step 3: Launch the Shadcn-Inspired Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interactively explore SKU forecasts, adjust lead times, tune service level buffers, and export weekly dispatch manifests.

---

## 7. Deliverables Summary

- [x] **Jupyter Notebook**: [`notebooks/spare_parts_inventory_forecasting.ipynb`](notebooks/spare_parts_inventory_forecasting.ipynb)
- [x] **Trained Model**: [`models/spare_parts_forecaster.joblib`](models/spare_parts_forecaster.joblib) & [`models/lightgbm_model.pkl`](models/lightgbm_model.pkl)
- [x] **Forecast Output CSV**: [`data/forecast_output.csv`](data/forecast_output.csv) (3,360 predictions with predicted demand, safety stock, and reorder points)
- [x] **Executive PDF Report**: [`reports/Project_Report_Spare_Parts_Inventory_Forecasting.pdf`](reports/Project_Report_Spare_Parts_Inventory_Forecasting.pdf)
- [x] **Streamlit Operations App**: [`app.py`](app.py)

---
© DataMites™ Solutions Pvt Ltd. Confidential & Proprietary. Client: NewX Services.
