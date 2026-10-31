"""
model.py
--------
Model definitions, pipeline constructors, and experiment execution routines
for crop yield forecasting under distribution shifts.
Supports:
- Approach A: Raw Yield Level
- Approach B: Trend-Aware Feature
- Approach C: Detrended Anomaly Modeling (In-sample OLS secular trend)
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from .preprocessing import (
    load_config,
    load_data,
    get_feature_groups,
    get_preprocessor,
    split_random,
    split_temporal,
    get_expanding_window_folds,
    get_geographic_loso_folds,
    fit_ols_yield_trend,
    compute_yield_anomalies,
    reconstruct_yield_from_anomalies,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Compute regression performance metrics: MAE, RMSE, and R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 4),
        "N": len(y_true)
    }


def get_models(random_seed: int = 42) -> Dict[str, Any]:
    """Instantiate standard suite of agricultural regression models."""
    models = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_seed, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed),
    }
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=random_seed, verbose=-1
        )
    return models


def run_baseline_experiment(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute baseline experiment using random 80/20 train/test split (IID assumption)."""
    if config is None:
        config = load_config()
    
    random_seed = config.get("random_seed", 42)
    _, _, numeric_feats = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    
    train_df, test_df = split_random(df, test_size=0.20, random_seed=random_seed)
    
    X_train = train_df[numeric_feats]
    y_train = train_df[target_col].values
    X_test = test_df[numeric_feats]
    y_test = test_df[target_col].values
    
    results = []
    models = get_models(random_seed=random_seed)
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        m = compute_metrics(y_test, preds)
        results.append({
            "Model": name,
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"],
            "N_test": m["N"]
        })
    return pd.DataFrame(results)


def run_temporal_shift_experiment(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Execute forward temporal shift experiment:
    Trained strictly on historical years (2000-2017), tested on future years (2018-2023).
    Compares:
      - Approach A: Raw Yield Level
      - Approach B: Trend-Aware Feature (year included as predictor)
      - Approach C: Detrended Anomaly (in-sample OLS trend subtraction and projection)
    """
    if config is None:
        config = load_config()
        
    random_seed = config.get("random_seed", 42)
    _, _, numeric_feats = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    year_col = config["data"]["year_col"]
    
    train_end = config.get("evaluation", {}).get("temporal_split", {}).get("train_end_year", 2017)
    test_start = config.get("evaluation", {}).get("temporal_split", {}).get("test_start_year", 2018)
    
    train_df, test_df = split_temporal(df, train_end_year=train_end, test_start_year=test_start, year_col=year_col)
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    results = []
    models = get_models(random_seed=random_seed)
    
    # --- Approach A: Raw Yield Level ---
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", model)
        ])
        pipe.fit(train_df[numeric_feats], y_train)
        preds = pipe.predict(test_df[numeric_feats])
        m = compute_metrics(y_test, preds)
        results.append({
            "Approach": "Approach A (Raw Yield)",
            "Model": name,
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"],
            "N_test": m["N"]
        })
        
    # --- Approach B: Trend-Aware Feature ---
    feats_with_year = numeric_feats + [year_col]
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", get_preprocessor(feats_with_year)),
            ("model", model)
        ])
        pipe.fit(train_df[feats_with_year], y_train)
        preds = pipe.predict(test_df[feats_with_year])
        m = compute_metrics(y_test, preds)
        results.append({
            "Approach": "Approach B (Trend-Aware Feature)",
            "Model": name,
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"],
            "N_test": m["N"]
        })
        
    # --- Approach C: Detrended Anomaly ---
    ols_trend = fit_ols_yield_trend(train_df, year_col=year_col, target_col=target_col)
    y_train_anomaly = compute_yield_anomalies(train_df, ols_trend, year_col=year_col, target_col=target_col)
    
    for name, model in models.items():
        pipe = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", model)
        ])
        pipe.fit(train_df[numeric_feats], y_train_anomaly)
        pred_anomalies = pipe.predict(test_df[numeric_feats])
        final_preds = reconstruct_yield_from_anomalies(ols_trend, test_df, pred_anomalies, year_col=year_col)
        m = compute_metrics(y_test, final_preds)
        results.append({
            "Approach": "Approach C (Detrended Anomaly)",
            "Model": name,
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"],
            "N_test": m["N"]
        })
        
    return pd.DataFrame(results)


def run_expanding_window_experiment(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute annual expanding-window forward-chaining evaluation (2016 to 2022)."""
    if config is None:
        config = load_config()
        
    random_seed = config.get("random_seed", 42)
    _, _, numeric_feats = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    year_col = config["data"]["year_col"]
    
    folds = get_expanding_window_folds(df, start_year=2015, end_year=2021, year_col=year_col)
    results = []
    
    for train_fold, test_fold, test_year in folds:
        y_train = train_fold[target_col].values
        y_test = test_fold[target_col].values
        
        # 1. Raw HistGradientBoosting
        pipe_raw = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
        ])
        pipe_raw.fit(train_fold[numeric_feats], y_train)
        pred_raw = pipe_raw.predict(test_fold[numeric_feats])
        m_raw = compute_metrics(y_test, pred_raw)
        
        # 2. Detrended HistGradientBoosting
        ols = fit_ols_yield_trend(train_fold, year_col=year_col, target_col=target_col)
        y_tr_anom = compute_yield_anomalies(train_fold, ols, year_col=year_col, target_col=target_col)
        
        pipe_det = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
        ])
        pipe_det.fit(train_fold[numeric_feats], y_tr_anom)
        pred_anom = pipe_det.predict(test_fold[numeric_feats])
        pred_det = reconstruct_yield_from_anomalies(ols, test_fold, pred_anom, year_col=year_col)
        m_det = compute_metrics(y_test, pred_det)
        
        results.append({
            "Test_Year": test_year,
            "Train_Years": f"2000-{test_year - 1}",
            "N_train": len(train_fold),
            "N_test": len(test_fold),
            "Raw_MAE": m_raw["MAE"],
            "Raw_R2": m_raw["R2"],
            "Detrended_MAE": m_det["MAE"],
            "Detrended_R2": m_det["R2"],
            "MAE_Improvement": round(m_raw["MAE"] - m_det["MAE"], 3)
        })
        
    return pd.DataFrame(results)


def run_geographic_loso_experiment(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute Leave-One-State-Out (LOSO) spatial cross-validation."""
    if config is None:
        config = load_config()
        
    random_seed = config.get("random_seed", 42)
    _, _, numeric_feats = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    year_col = config["data"]["year_col"]
    state_col = config["data"]["state_col"]
    
    folds = get_geographic_loso_folds(df, state_col=state_col)
    results = []
    
    for train_fold, test_fold, held_out_state in folds:
        y_train = train_fold[target_col].values
        y_test = test_fold[target_col].values
        
        # Raw model
        pipe_raw = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
        ])
        pipe_raw.fit(train_fold[numeric_feats], y_train)
        pred_raw = pipe_raw.predict(test_fold[numeric_feats])
        m_raw = compute_metrics(y_test, pred_raw)
        
        # Detrended model
        ols = fit_ols_yield_trend(train_fold, year_col=year_col, target_col=target_col)
        y_tr_anom = compute_yield_anomalies(train_fold, ols, year_col=year_col, target_col=target_col)
        pipe_det = Pipeline([
            ("prep", get_preprocessor(numeric_feats)),
            ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
        ])
        pipe_det.fit(train_fold[numeric_feats], y_tr_anom)
        pred_anom = pipe_det.predict(test_fold[numeric_feats])
        pred_det = reconstruct_yield_from_anomalies(ols, test_fold, pred_anom, year_col=year_col)
        m_det = compute_metrics(y_test, pred_det)
        
        results.append({
            "Held_Out_State": held_out_state,
            "N_test": len(test_fold),
            "Raw_MAE": m_raw["MAE"],
            "Raw_R2": m_raw["R2"],
            "Detrended_MAE": m_det["MAE"],
            "Detrended_R2": m_det["R2"],
            "MAE_Improvement": round(m_raw["MAE"] - m_det["MAE"], 3)
        })
        
    return pd.DataFrame(results)


def run_feature_ablation_experiment(df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Execute feature group ablation study under temporal holdout."""
    if config is None:
        config = load_config()
        
    random_seed = config.get("random_seed", 42)
    weather_feats, soil_feats, all_numeric = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    year_col = config["data"]["year_col"]
    
    train_df, test_df = split_temporal(df, year_col=year_col)
    
    ols = fit_ols_yield_trend(train_df, year_col=year_col, target_col=target_col)
    y_tr_anom = compute_yield_anomalies(train_df, ols, year_col=year_col, target_col=target_col)
    y_test = test_df[target_col].values
    
    july_feats = [f for f in all_numeric if "july" in f.lower()]
    
    feature_sets = {
        "All Features (Weather + Soil)": all_numeric,
        "Weather Only (No Soil)": weather_feats,
        "Soil Only (No Weather)": soil_feats,
        "No July Critical Features": [f for f in all_numeric if f not in july_feats]
    }
    
    results = []
    for feat_name, feats in feature_sets.items():
        pipe = Pipeline([
            ("prep", get_preprocessor(feats)),
            ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
        ])
        pipe.fit(train_df[feats], y_tr_anom)
        pred_anom = pipe.predict(test_df[feats])
        pred_final = reconstruct_yield_from_anomalies(ols, test_df, pred_anom, year_col=year_col)
        m = compute_metrics(y_test, pred_final)
        results.append({
            "Feature_Set": feat_name,
            "N_features": len(feats),
            "Test_MAE": m["MAE"],
            "Test_RMSE": m["RMSE"],
            "Test_R2": m["R2"]
        })
        
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Executing full model experimental pipeline...")
    df_data = load_data()
    
    print("\n1. Running Baseline Experiments (Random Split)...")
    base_df = run_baseline_experiment(df_data)
    print(base_df.to_string(index=False))
    
    print("\n2. Running Temporal Shift Experiments (2018-2023 Holdout)...")
    temp_df = run_temporal_shift_experiment(df_data)
    print(temp_df.to_string(index=False))
    
    print("\n3. Running Expanding Window Experiments...")
    exp_df = run_expanding_window_experiment(df_data)
    print(exp_df.to_string(index=False))
    
    print("\n4. Running Geographic LOSO Experiments...")
    geo_df = run_geographic_loso_experiment(df_data)
    print(geo_df.to_string(index=False))
    
    print("\n5. Running Feature Ablation Experiments...")
    abl_df = run_feature_ablation_experiment(df_data)
    print(abl_df.to_string(index=False))
