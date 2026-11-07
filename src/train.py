"""
Model Training, Cross-Validation, and Inventory Evaluation Pipeline
AI Client Project 1 — Spare Parts Inventory Forecasting (NewX Services)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

def calculate_inventory_metrics(y_true, y_pred):
    """
    Computes inventory KPIs: Overstock Rate, Stockout Rate, Overstock Units, Stockout Units, WAPE.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.maximum(0, np.array(y_pred, dtype=float)) # Demand cannot be negative
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # WAPE (Weighted Absolute Percentage Error = sum(|y - y_hat|) / sum(y))
    total_actual = np.sum(y_true)
    wape = (np.sum(np.abs(y_true - y_pred)) / total_actual * 100) if total_actual > 0 else 0.0
    
    # Overstock: predicted > actual
    overstock_mask = y_pred > y_true
    overstock_rate = np.mean(overstock_mask) * 100.0
    overstock_units = np.sum(np.maximum(0, y_pred - y_true))
    
    # Stockout: predicted < actual
    stockout_mask = y_pred < y_true
    stockout_rate = np.mean(stockout_mask) * 100.0
    stockout_units = np.sum(np.maximum(0, y_true - y_pred))
    
    # Exact match: predicted == actual
    exact_match_rate = np.mean(np.isclose(y_pred, y_true, atol=0.5)) * 100.0
    
    return {
        'MAE': round(float(mae), 4),
        'RMSE': round(float(rmse), 4),
        'WAPE_pct': round(float(wape), 2),
        'Overstock_Rate_pct': round(float(overstock_rate), 2),
        'Stockout_Rate_pct': round(float(stockout_rate), 2),
        'Overstock_Units': round(float(overstock_units), 1),
        'Stockout_Units': round(float(stockout_units), 1),
        'Exact_Match_pct': round(float(exact_match_rate), 2)
    }

def train_and_evaluate(featured_csv='data/processed/featured_demand_dataset.csv'):
    df = pd.read_csv(featured_csv)
    df['week_start'] = pd.to_datetime(df['week_start'])
    
    # Temporal Train / Test Split
    # Last 14 weeks (~3.5 months) as held-out test set
    unique_weeks = sorted(df['week_start'].unique())
    test_horizon_weeks = 14
    split_date = unique_weeks[-test_horizon_weeks]
    
    train_mask = df['week_start'] < split_date
    test_mask = df['week_start'] >= split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Total Weeks: {len(unique_weeks)} | Split Date: {split_date.strftime('%Y-%m-%d')}")
    print(f"Train samples: {len(train_df):,} | Test samples: {len(test_df):,}")
    
    # Feature definitions
    categorical_cols = ['service_centre', 'part_name', 'subsystem', 'km_bin', 'demand_pattern']
    feature_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_8', 'lag_12',
        'rolling_mean_4', 'rolling_std_4', 'rolling_max_4', 'rolling_min_4',
        'rolling_mean_8', 'rolling_std_8',
        'rolling_mean_12', 'rolling_std_12',
        'expanding_mean', 'demand_momentum',
        'month', 'quarter', 'week_of_year',
        'sin_week', 'cos_week', 'sin_month', 'cos_month',
        'sc_traffic_lag1', 'adi', 'cv2'
    ]
    
    # One-hot / category encoding for tree models
    for col in categorical_cols:
        train_df[col] = train_df[col].astype('category')
        test_df[col] = test_df[col].astype('category')
        
    all_features = feature_cols + categorical_cols
    
    X_train = train_df[all_features].copy()
    y_train = train_df['demand'].values
    
    X_test = test_df[all_features].copy()
    y_test = test_df['demand'].values
    
    results = {}
    test_predictions = test_df.copy()
    
    # 1. Naive Baseline: Last week's demand (lag_1)
    naive_pred = test_df['lag_1'].values
    results['Naive (Last Week)'] = calculate_inventory_metrics(y_test, naive_pred)
    test_predictions['pred_naive'] = naive_pred
    
    # 2. Moving Average Baseline: 4-week rolling mean
    ma_pred = test_df['rolling_mean_4'].values
    results['Moving Average (4-Week)'] = calculate_inventory_metrics(y_test, ma_pred)
    test_predictions['pred_ma4'] = ma_pred
    
    # 3. LightGBM Regressor
    print("\nTraining LightGBM Regressor...")
    lgb_train_data = lgb.Dataset(X_train, label=y_train)
    lgb_test_data = lgb.Dataset(X_test, label=y_test, reference=lgb_train_data)
    
    lgb_params = {
        'objective': 'poisson', # Poisson / Tweedie ideal for count / intermittent demand
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'random_state': 42,
        'verbose': -1
    }
    
    lgb_model = lgb.train(
        lgb_params,
        lgb_train_data,
        num_boost_round=400,
        valid_sets=[lgb_test_data]
    )
    
    lgb_pred = lgb_model.predict(X_test)
    results['LightGBM (Poisson)'] = calculate_inventory_metrics(y_test, lgb_pred)
    test_predictions['pred_lightgbm'] = lgb_pred
    
    # 4. XGBoost Regressor
    print("Training XGBoost Regressor...")
    xgb_train = X_train.copy()
    xgb_test = X_test.copy()
    # XGBoost category support
    for c in categorical_cols:
        xgb_train[c] = xgb_train[c].astype('category')
        xgb_test[c] = xgb_test[c].astype('category')
        
    xgb_model = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        enable_categorical=True,
        random_state=42
    )
    xgb_model.fit(xgb_train, y_train)
    xgb_pred = xgb_model.predict(xgb_test)
    results['XGBoost (Poisson)'] = calculate_inventory_metrics(y_test, xgb_pred)
    test_predictions['pred_xgboost'] = xgb_pred
    
    # 5. Random Forest Regressor
    print("Training Random Forest Regressor...")
    # Label encode categoricals for sklearn RF
    rf_train = X_train.copy()
    rf_test = X_test.copy()
    for c in categorical_cols:
        rf_train[c] = rf_train[c].cat.codes
        rf_test[c] = rf_test[c].cat.codes
        
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(rf_train, y_train)
    rf_pred = rf_model.predict(rf_test)
    results['Random Forest'] = calculate_inventory_metrics(y_test, rf_pred)
    test_predictions['pred_rf'] = rf_pred
    
    # Compare against Naive Baseline
    naive_stockout_units = results['Naive (Last Week)']['Stockout_Units']
    naive_overstock_units = results['Naive (Last Week)']['Overstock_Units']
    
    print("\n================== BENCHMARK MODEL EVALUATION ==================")
    summary_rows = []
    for m_name, m_metrics in results.items():
        so_red = ((naive_stockout_units - m_metrics['Stockout_Units']) / naive_stockout_units) * 100
        os_red = ((naive_overstock_units - m_metrics['Overstock_Units']) / naive_overstock_units) * 100
        summary_rows.append({
            'Model': m_name,
            'MAE': m_metrics['MAE'],
            'RMSE': m_metrics['RMSE'],
            'WAPE (%)': m_metrics['WAPE_pct'],
            'Overstock Rate (%)': m_metrics['Overstock_Rate_pct'],
            'Stockout Rate (%)': m_metrics['Stockout_Rate_pct'],
            'Overstock Reduction (%)': round(os_red, 2),
            'Stockout Reduction (%)': round(so_red, 2),
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    
    # Pick Champion Model (LightGBM)
    champion_name = 'LightGBM'
    champion_pred = lgb_pred
    test_predictions['predicted_demand'] = np.round(champion_pred, 2)
    
    # Calculate Safety Stock (SS) & Reorder Point (ROP) for Just-In-Time (JIT) operations
    # Service level Z = 1.65 (95% JIT target), Lead time L = 1 week
    lead_time_weeks = 1.0
    z_score = 1.65
    
    test_predictions['demand_error_std'] = np.std(y_test - champion_pred)
    # Series specific volatility
    test_predictions['safety_stock'] = np.ceil(
        z_score * np.sqrt(lead_time_weeks) * np.maximum(0.5, test_predictions['rolling_std_4'])
    ).astype(int)
    
    test_predictions['reorder_point'] = np.ceil(
        test_predictions['predicted_demand'] * lead_time_weeks + test_predictions['safety_stock']
    ).astype(int)
    
    # JIT Inventory Risk Status Flag
    conditions = [
        test_predictions['predicted_demand'] > 5,
        test_predictions['predicted_demand'] > 2,
        test_predictions['predicted_demand'] > 0
    ]
    choices = ['High Demand Alert', 'Moderate Replenishment', 'Buffer Hold']
    test_predictions['inventory_action'] = np.select(conditions, choices, default='No Order Needed')
    
    # Save Champion Model Artifacts
    os.makedirs('models', exist_ok=True)
    model_bundle = {
        'champion_model': lgb_model,
        'model_name': 'LightGBM-Poisson-Forecaster',
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'all_features': all_features,
        'evaluation_summary': summary_rows
    }
    joblib.dump(model_bundle, 'models/spare_parts_forecaster.joblib')
    joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
    
    # Save Metadata JSON
    with open('models/model_metadata.json', 'w') as f:
        json.dump({
            'champion': 'LightGBM Regressor (Poisson)',
            'training_weeks': int(len(unique_weeks) - test_horizon_weeks),
            'test_weeks': int(test_horizon_weeks),
            'split_date': str(split_date),
            'metrics_comparison': summary_rows
        }, f, indent=2)
        
    # Feature Importance
    importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': lgb_model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    importance_df.to_csv('models/feature_importance.csv', index=False)
    
    # Generate Deliverable Forecast Output CSV
    # part_name x service_centre x week -> predicted_demand
    forecast_cols = [
        'part_name', 'service_centre', 'week_start', 'demand', 'predicted_demand',
        'safety_stock', 'reorder_point', 'inventory_action', 'subsystem', 'demand_pattern'
    ]
    forecast_output = test_predictions[forecast_cols].copy()
    forecast_output.rename(columns={'demand': 'actual_demand'}, inplace=True)
    forecast_output.to_csv('data/forecast_output.csv', index=False)
    print(f"\nDeliverable Forecast CSV saved to data/forecast_output.csv: {len(forecast_output):,} predictions.")
    
    summary_df.to_csv('models/model_benchmark_results.csv', index=False)
    return summary_df, forecast_output

if __name__ == '__main__':
    summary, forecast = train_and_evaluate()
