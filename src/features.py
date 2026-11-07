"""
Feature Engineering Engine for Spare Parts Demand Forecasting
AI Client Project 1 — Spare Parts Inventory Forecasting (NewX Services)
"""

import numpy as np
import pandas as pd

def compute_intermittency_metrics(series):
    """
    Computes Average Demand Interval (ADI) and Squared Coefficient of Variation (CV^2).
    """
    non_zero = series[series > 0]
    if len(non_zero) < 2:
        return 999.0, 999.0, 'Lumpy'
    
    # Non-zero intervals
    idx = np.where(series > 0)[0]
    intervals = np.diff(idx)
    adi = np.mean(intervals) if len(intervals) > 0 else len(series)
    
    # CV^2
    mean_val = np.mean(non_zero)
    std_val = np.std(non_zero)
    cv2 = (std_val / mean_val) ** 2 if mean_val > 0 else 0
    
    # Categorization
    if adi < 1.32 and cv2 < 0.49:
        cat = 'Smooth'
    elif adi >= 1.32 and cv2 < 0.49:
        cat = 'Intermittent'
    elif adi < 1.32 and cv2 >= 0.49:
        cat = 'Erratic'
    else:
        cat = 'Lumpy'
    return round(float(adi), 2), round(float(cv2), 2), cat

def engineer_features(panel_df):
    """
    Generates lag features, rolling windows, seasonal variables, and vehicle context.
    """
    df = panel_df.copy()
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Ensure sorted order
    df = df.sort_values(['service_centre', 'part_name', 'week_start']).reset_index(drop=True)
    group = df.groupby(['service_centre', 'part_name'])
    
    # 1. Demand Lags (Shifted so no data leakage occurs)
    for lag in [1, 2, 3, 4, 8, 12]:
        df[f'lag_{lag}'] = group['demand'].shift(lag)
        
    # 2. Rolling Aggregations on shifted demand (excluding current week)
    shifted = group['demand'].shift(1)
    
    # 4-week window
    df['rolling_mean_4'] = group['demand'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    df['rolling_std_4'] = group['demand'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).std()).fillna(0)
    df['rolling_max_4'] = group['demand'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).max())
    df['rolling_min_4'] = group['demand'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).min())
    
    # 8-week window
    df['rolling_mean_8'] = group['demand'].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
    df['rolling_std_8'] = group['demand'].transform(lambda x: x.shift(1).rolling(8, min_periods=1).std()).fillna(0)
    
    # 12-week window (Quarterly)
    df['rolling_mean_12'] = group['demand'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).mean())
    df['rolling_std_12'] = group['demand'].transform(lambda x: x.shift(1).rolling(12, min_periods=1).std()).fillna(0)
    
    # Expanding Mean
    df['expanding_mean'] = group['demand'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    
    # Demand Trend Indicator: ratio of recent 4-week mean to 12-week mean
    df['demand_momentum'] = (df['rolling_mean_4'] + 1e-4) / (df['rolling_mean_12'] + 1e-4)
    
    # 3. Calendar & Seasonality
    df['month'] = df['week_start'].dt.month
    df['quarter'] = df['week_start'].dt.quarter
    df['week_of_year'] = df['week_start'].dt.isocalendar().week.astype(int)
    
    # Cyclical sin/cos encodings
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 4. Service Centre & Mileage Context
    # Total centre traffic lag
    sc_traffic = df.groupby(['week_start', 'service_centre'])['demand'].sum().reset_index()
    sc_traffic.rename(columns={'demand': 'sc_total_weekly_demand'}, inplace=True)
    sc_traffic['sc_traffic_lag1'] = sc_traffic.groupby('service_centre')['sc_total_weekly_demand'].shift(1)
    df = pd.merge(df, sc_traffic[['week_start', 'service_centre', 'sc_traffic_lag1']], on=['week_start', 'service_centre'], how='left')
    
    # Mileage Bins (KM interval bins)
    km_bins = [-1, 10000, 25000, 45000, 75000, 200000]
    km_labels = ['0-10k_Fresh', '10k-25k_MidLife', '25k-45k_Mature', '45k-75k_HighWear', '75k+_EndLife']
    df['km_bin'] = pd.cut(df['avg_km'], bins=km_bins, labels=km_labels).astype(str)
    
    # 5. Intermittency Profile per Series
    intermittency_records = []
    for (sc, part), sub in df.groupby(['service_centre', 'part_name']):
        adi, cv2, cat = compute_intermittency_metrics(sub['demand'].values)
        intermittency_records.append({
            'service_centre': sc,
            'part_name': part,
            'adi': adi,
            'cv2': cv2,
            'demand_pattern': cat
        })
    inter_df = pd.DataFrame(intermittency_records)
    df = pd.merge(df, inter_df, on=['service_centre', 'part_name'], how='left')
    
    # Drop warm-up weeks with NaN in critical lags (first 12 weeks used as lag burn-in)
    feature_df = df.dropna(subset=['lag_12']).copy().reset_index(drop=True)
    print(f"Feature dataset built: {feature_df.shape[0]:,} rows, {feature_df.shape[1]} columns.")
    print("Demand pattern distribution across SKU x SC pairs:")
    print(inter_df['demand_pattern'].value_counts())
    
    feature_df.to_csv('data/processed/featured_demand_dataset.csv', index=False)
    inter_df.to_csv('data/processed/intermittency_profile.csv', index=False)
    return feature_df, inter_df

if __name__ == '__main__':
    panel = pd.read_csv('data/processed/weekly_demand_panel.csv')
    feat_df, inter_df = engineer_features(panel)
