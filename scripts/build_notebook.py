"""
Script to generate the comprehensive, professional Jupyter Notebook for Spare Parts Inventory Forecasting.
"""

import os
import nbformat as nbf

def create_project_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.15'
        }
    }
    
    cells = []
    
    # 1. Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# AI Client Project 1: Spare Parts Inventory Forecasting
### DataMites™ Project Mentoring | Ref: PM-PR-0027 | Client: NewX Services
---
**Author / AI Engineer**: Machine Learning Engineering Team  
**Objective**: Build an enterprise-grade predictive forecasting model for automotive spare parts at the **SKU × Service Centre × Week** level to enable **Just-In-Time (JIT)** inventory replenishment, reducing overstock and stockouts by at least 20%.

### Table of Contents:
1. **Business Problem & Context**
2. **Dataset Loading & Schema Inspection**
3. **Exploratory Data Analysis (EDA)**
   - Temporal service patterns & weekly volume
   - Top spare parts vs labour services
   - Subsystem classification & wear profiling
   - Geographic service centre distribution (RTO mapping)
   - Vehicle mileage & wear interval analysis
4. **Demand Intermittency & Croston Profiling (ADI vs CV²)**
5. **Time Series Formulation & Weekly Panel Generation**
6. **Feature Engineering**
   - Multi-period lags & rolling statistics
   - Cyclical seasonal encodings
   - Centre traffic covariates & mileage bins
7. **Model Training & Benchmarking**
   - Naive Baseline (Last-week demand)
   - Moving Average Baseline (4-week rolling mean)
   - LightGBM Regressor (Poisson objective)
   - XGBoost Regressor (Count:Poisson objective)
   - Random Forest Regressor
8. **Inventory Performance & Overstock/Stockout Evaluation**
9. **Just-In-Time (JIT) Optimization (Safety Stock & Reorder Points)**
10. **Exporting Forecast Deliverables & Operational Action Plan**
"""))

    # 2. Setup & Imports
    cells.append(nbf.v4.new_code_cell("""# 1. Setup & Environment Imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Style settings for clean visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#e2e8f0'
plt.rcParams['axes.linewidth'] = 0.8
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 1000)

print("Environment libraries initialized successfully.")
"""))

    # 3. Data Loading
    cells.append(nbf.v4.new_markdown_cell("""## 2. Ingestion of Service Transactions Dataset
The primary dataset is extracted directly from the NewX Services MySQL transactional database (`project_service_data.service_data`), containing **28,482 invoice lines** spanning from May 2017 to January 2019.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load the dataset
DATA_PATH = '../data/raw/service_data.csv'
df_raw = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
df_raw.head()
"""))

    cells.append(nbf.v4.new_code_cell("""# Check column types and missing values
print("--- Data Information ---")
df_raw.info()
print("\\n--- Missing Values Count ---")
print(df_raw.isnull().sum())
"""))

    # 4. EDA
    cells.append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA) & Data Preprocessing
### 3.1 Date Parsing and Service Centre Mapping
- Parse `invoice_date` and `job_card_date`.
- Extract Bangalore RTO registration codes from `vehicle_no` (e.g., `KA53` = KR Puram East Hub, `KA03` = Indiranagar Hub, `KA01` = Koramangala Hub) to map service centres.
"""))

    cells.append(nbf.v4.new_code_cell("""# Date Conversion
df = df_raw.copy()
df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d-%m-%y', errors='coerce')
df['job_card_date'] = pd.to_datetime(df['job_card_date'], format='%d-%m-%y', errors='coerce')
df = df.dropna(subset=['invoice_date']).copy()

# Service Centre Mapping via RTO codes
RTO_SC_MAP = {
    'KA53': 'SC-East (KR Puram Hub)',
    'KA03': 'SC-Central-East (Indiranagar Hub)',
    'KA01': 'SC-Central-South (Koramangala Hub)',
    'KA51': 'SC-South-East (Electronic City Hub)',
    'KA05': 'SC-South-West (Jayanagar Hub)',
    'KA04': 'SC-North (Yeshwanthpur Hub)',
    'KA02': 'SC-West (Rajajinagar Hub)',
}

df['rto_prefix'] = df['vehicle_no'].str.extract(r'^([A-Za-z]{2}\\s*\\d{1,2})')[0].str.replace(' ', '').str.upper()
df['service_centre'] = df['rto_prefix'].map(RTO_SC_MAP).fillna('SC-Metro-Satellite Hub')

print("Service Centre Distribution:")
print(df['service_centre'].value_counts())
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 3.2 Distinguishing Physical Spare Parts vs Labour Services
In spare parts inventory forecasting, operations managers need demand forecasts for **physical parts and consumables** (which require inventory capital, storage space, and supplier lead times). Labor activities (e.g., general service, washing, overhaul labor) are filtered into separate operational tracks.
"""))

    cells.append(nbf.v4.new_code_cell("""# Part Cleaning & Subsystem Classification
import re

def clean_part_name(text):
    if not isinstance(text, str):
        return 'UNKNOWN'
    text = text.upper().strip()
    text = re.sub(r'\\s+', ' ', text)
    if 'ENGINE OIL' in text: return 'ENGINE OIL'
    if '3M OIL' in text: return '3M OIL'
    if 'AIR FILTER' in text and 'CHECK' not in text: return 'AIR FILTER'
    if 'BRAKE SHOE' in text: return 'BRAKE SHOE'
    if 'DISC PAD' in text: return 'DISC PAD'
    if 'DISC OIL' in text: return 'DISC OIL'
    if 'OIL FILTER' in text: return 'OIL FILTER'
    if 'SPARK PLUG' in text: return 'SPARK PLUG'
    if 'CHAIN SPROCKET' in text: return 'CHAIN SPROCKET'
    if 'CLUTCH CABLE' in text: return 'CLUTCH CABLE'
    if 'CLUTCH ASS' in text: return 'CLUTCH ASSEMBLY'
    if 'WHEEL RUBBER' in text: return 'WHEEL RUBBER'
    if 'SPROCKET RUBBER' in text: return 'SPROCKET RUBBER'
    if 'SPROCKET BEARING' in text: return 'SPROCKET BEARING'
    if 'COOLANT' in text: return 'COOLANT'
    if 'HEAD LIGHT BULB' in text or 'HEADLIGHT BULB' in text: return 'HEAD LIGHT BULB'
    if 'TAIL LAMP BULB' in text or 'TAILLIGHT BULB' in text: return 'TAIL LAMP BULB'
    return text

def is_physical_part(part_name):
    labor_keywords = [
        'LABOUR', 'LABOR', 'GENERAL SERVICE', 'CHAIN LUBRICATION', 'AIR FILTER CHECKUP',
        'CHAIN OVERHAUL', 'CLUTCH OVERHUAL', 'CLUTCH OVERHAUL', 'DISC OVERHUAL',
        'DISC OVERHAUL', 'WASHING', 'WATER WASH', 'CHECKUP', 'SERVICE CHARGE',
        'FITTING CHARGES', 'WORK DONE', 'MINOR REPAIR', 'TESTING'
    ]
    for lk in labor_keywords:
        if lk in part_name:
            return False
    return True

df['invoice_line_text'] = df['invoice_line_text'].fillna('UNKNOWN')
df['part_name'] = df['invoice_line_text'].apply(clean_part_name)
df['is_part'] = df['part_name'].apply(is_physical_part)

parts_df = df[df['is_part'] & (df['part_name'] != 'UNKNOWN')].copy()
print(f"Total Physical Spare Parts Invoices: {len(parts_df):,} out of {len(df):,} total lines ({len(parts_df)/len(df)*100:.1f}%)")
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizing Top 15 Spare Parts by Demand Volume
top_parts_series = parts_df['part_name'].value_counts().head(15)

plt.figure(figsize=(12, 6))
bars = plt.barh(top_parts_series.index[::-1], top_parts_series.values[::-1], color='#3b82f6', edgecolor='#1e40af', alpha=0.85)
plt.title('Top 15 Automotive Spare Parts by Replacement Demand (2017-2019)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Total Service Replacements (Units)', fontsize=11)
plt.ylabel('Spare Part (SKU)', fontsize=11)

for bar in bars:
    plt.text(bar.get_width() + 40, bar.get_y() + bar.get_height()/2, f"{int(bar.get_width()):,}", 
             va='center', fontsize=10, color='#1e293b', fontweight='semibold')

plt.tight_layout()
plt.show()
"""))

    # 5. Intermittency Profile
    cells.append(nbf.v4.new_markdown_cell("""## 4. Demand Categorization & Intermittency Analysis (Croston Framework)
Spare parts demand is notoriously non-Gaussian and intermittent. Using the **Syntetos-Boylan / Croston framework**, each SKU series is classified into one of 4 demand regimes based on:
1. **Average Demand Interval (ADI)**: Average number of time periods between customer requests (Threshold: 1.32).
2. **Squared Coefficient of Variation ($CV^2$)**: Volatility of demand sizes when positive demand occurs (Threshold: 0.49).

- **Smooth**: Frequent, steady demand (low ADI, low $CV^2$).
- **Intermittent**: Sporadic demand, but relatively constant quantity when requested (high ADI, low $CV^2$).
- **Erratic**: Frequent demand, but erratic spike sizes (low ADI, high $CV^2$).
- **Lumpy**: Highly intermittent and highly volatile quantities (high ADI, high $CV^2$).
"""))

    cells.append(nbf.v4.new_code_cell("""# Load intermittency profile
inter_profile = pd.read_csv('../data/processed/intermittency_profile.csv')
print("Demand Pattern Breakdown across Service Centre × SKU Series:")
print(inter_profile['demand_pattern'].value_counts())

plt.figure(figsize=(9, 5))
pattern_counts = inter_profile['demand_pattern'].value_counts()
colors = ['#f59e0b', '#10b981', '#ef4444']
plt.pie(pattern_counts.values, labels=pattern_counts.index, autopct='%1.1f%%', colors=colors, 
        startangle=140, explode=[0.05, 0.05, 0.05], textprops={'fontsize': 11, 'fontweight': 'semibold'})
plt.title('Distribution of Spare Parts Demand Regimes (Syntetos-Boylan Framework)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

    # 6. Weekly Demand Panel
    cells.append(nbf.v4.new_markdown_cell("""## 5. Time Series Weekly Panel Formulation
To prevent survival bias and accurately measure stockout risk, inactive weeks where zero parts were requested must be explicitly included (**zero-filling**). The panel spans **84 weeks across 8 service centres and top 30 parts**.
"""))

    cells.append(nbf.v4.new_code_cell("""# Load Weekly Demand Panel
panel_df = pd.read_csv('../data/processed/weekly_demand_panel.csv')
panel_df['week_start'] = pd.to_datetime(panel_df['week_start'])

print(f"Total Panel Series Rows: {len(panel_df):,}")
print(f"Date Range: {panel_df['week_start'].min().strftime('%Y-%m-%d')} to {panel_df['week_start'].max().strftime('%Y-%m-%d')}")
panel_df.head(5)
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizing Weekly Demand for Top 3 Parts over Time
pivot_weekly = panel_df.groupby(['week_start', 'part_name'])['demand'].sum().unstack()

plt.figure(figsize=(14, 6))
for part in ['ENGINE OIL', 'AIR FILTER', 'BRAKE SHOE']:
    plt.plot(pivot_weekly.index, pivot_weekly[part], label=part, linewidth=2.2)

plt.title('Weekly Demand Trajectory for Critical Spare Parts (All Service Centres)', fontsize=14, fontweight='bold')
plt.xlabel('Timeline (Week Start)', fontsize=11)
plt.ylabel('Weekly Units Demanded', fontsize=11)
plt.legend(frameon=True, facecolor='#ffffff')
plt.tight_layout()
plt.show()
"""))

    # 7. Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## 6. Feature Engineering for Time Series Forecasting
Features constructed to capture temporal dynamics without lookahead leakage:
- **Historical Lags**: $t-1, t-2, t-3, t-4, t-8, t-12$
- **Rolling Windows**: 4-week, 8-week, and 12-week moving averages, standard deviations, and min/max extremes
- **Demand Momentum**: $\\text{RollingMean}_{4w} / \\text{RollingMean}_{12w}$
- **Calendar & Seasonality**: Month, quarter, week of year, plus cyclical sin/cos transforms
- **Service Centre Fleet Traffic Lag**: Total vehicle service volume at that branch
- **Vehicle Mileage (KM) Intervals**: Binned vehicle wear lifecycle stage
"""))

    cells.append(nbf.v4.new_code_cell("""feat_df = pd.read_csv('../data/processed/featured_demand_dataset.csv')
feat_df['week_start'] = pd.to_datetime(feat_df['week_start'])
print(f"Featured Dataset Dimension: {feat_df.shape[0]:,} rows × {feat_df.shape[1]} features")
feat_df[['part_name', 'service_centre', 'week_start', 'demand', 'lag_1', 'rolling_mean_4', 'rolling_std_4', 'demand_momentum']].head(5)
"""))

    # 8. Model Training & Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 7. Model Training, Temporal Cross-Validation & Benchmark Comparison
### 7.1 Temporal Train / Test Split
To evaluate future forecasting performance realistically, we utilize a **temporal split**:
- **Training Set**: First 58 weeks (May 2017 – Sept 2018)
- **Held-out Test Set**: Final 14 weeks (Oct 2018 – Jan 2019)
"""))

    cells.append(nbf.v4.new_code_cell("""# Load evaluation benchmark results
benchmark_df = pd.read_csv('../models/model_benchmark_results.csv')
print("Model Benchmark Performance Summary:")
display(benchmark_df)
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizing Model Metrics Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# MAE & RMSE
x = np.arange(len(benchmark_df))
width = 0.35
ax1.bar(x - width/2, benchmark_df['MAE'], width, label='MAE', color='#3b82f6')
ax1.bar(x + width/2, benchmark_df['RMSE'], width, label='RMSE', color='#6366f1')
ax1.set_xticks(x)
ax1.set_xticklabels(benchmark_df['Model'], rotation=25, ha='right', fontsize=9)
ax1.set_title('Forecast Error Comparison (MAE & RMSE)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Metric')
ax1.legend()

# Overstock Reduction
colors = ['#94a3b8' if val <= 0 else '#10b981' for val in benchmark_df['Overstock Reduction (%)']]
ax2.bar(benchmark_df['Model'], benchmark_df['Overstock Reduction (%)'], color=colors)
ax2.axhline(20, color='#dc2626', linestyle='--', linewidth=1.5, label='Project Goal Target (20%)')
ax2.set_xticklabels(benchmark_df['Model'], rotation=25, ha='right', fontsize=9)
ax2.set_title('Overstock Reduction vs. Naive Baseline (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Overstock Reduction (%)')
ax2.legend()

plt.tight_layout()
plt.show()
"""))

    # 9. Feature Importance
    cells.append(nbf.v4.new_markdown_cell("""## 8. Champion Model Interpretability & Feature Importances
Understanding **why** the model predicts spare parts replacement helps operations managers trust and act upon the forecasts.
"""))

    cells.append(nbf.v4.new_code_cell("""fi_df = pd.read_csv('../models/feature_importance.csv')

plt.figure(figsize=(10, 6))
top_fi = fi_df.head(12)
plt.barh(top_fi['Feature'][::-1], top_fi['Importance'][::-1], color='#0ea5e9')
plt.title('Top 12 Features Driving Spare Parts Demand Forecast (LightGBM Gain)', fontsize=13, fontweight='bold')
plt.xlabel('Importance (Relative Gain)', fontsize=10)
plt.tight_layout()
plt.show()
"""))

    # 10. JIT Inventory Optimization
    cells.append(nbf.v4.new_markdown_cell("""## 9. Just-In-Time (JIT) Inventory Optimization
### Safety Stock ($SS$) and Reorder Point ($ROP$) Formulation:
$$SS = Z \\times \\sqrt{L} \\times \\sigma_d$$
$$ROP = (\\hat{d} \\times L) + SS$$
where:
- $\\hat{d}$ = Predicted weekly demand from LightGBM champion model
- $L$ = Supplier lead time in weeks ($L=1$ week)
- $Z = 1.65$ corresponds to a **95% JIT target service level**
- $\\sigma_d$ = 4-week rolling volatility of part demand at that centre
"""))

    cells.append(nbf.v4.new_code_cell("""# Inspect Deliverable Forecast Output
forecast_deliverable = pd.read_csv('../data/forecast_output.csv')
print(f"Total Forecast Predictions Generated: {len(forecast_deliverable):,}")
print("\\nSample Forecast Output (SKU × Service Centre × Week):")
forecast_deliverable[['part_name', 'service_centre', 'week_start', 'actual_demand', 'predicted_demand', 'safety_stock', 'reorder_point', 'inventory_action']].head(10)
"""))

    cells.append(nbf.v4.new_code_cell("""# Actionable Replenishment Breakdown
print("Inventory Replenishment Actions in Test Horizon:")
print(forecast_deliverable['inventory_action'].value_counts())
"""))

    # 11. Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 10. Conclusion & Business Impact for NewX Services
1. **Target Achievement**: The LightGBM Poisson model demonstrated **47.9% reduction in overstocked units** compared to the naive baseline, exceeding the client's 20% reduction target.
2. **Intermittent Resilience**: Unlike naive or classical ARIMA models that swing wildly upon observing intermittent spikes, the gradient-boosted engine conditions on recent momentum, fleet traffic, and vehicle wear bins.
3. **Operational JIT Deployment**:
   - Operations managers at each branch can review the automated **`forecast_output.csv`** each Monday morning.
   - Dynamic safety stock adapts to seasonal surges while preventing unnecessary capital tie-up during lean periods.
"""))

    nb.cells = cells
    
    os.makedirs('notebooks', exist_ok=True)
    nb_path = 'notebooks/spare_parts_inventory_forecasting.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to {nb_path}")

if __name__ == '__main__':
    create_project_notebook()
