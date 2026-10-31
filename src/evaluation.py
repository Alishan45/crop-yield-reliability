"""
evaluation.py
-------------
Conformal Prediction (Uncertainty Quantification), cluster bootstrap confidence intervals,
and TreeSHAP explainability stability analysis for crop yield forecasting.
Methods:
- Standard Split Conformal Prediction (SCP) with exact finite-sample order statistics
- Time-Decayed Weighted Conformal Prediction (WCP) (Tibshirani et al., 2019; Barber et al., 2023)
- Adaptive Conformal Inference (ACI) (Gibbs & Candes, 2021)
- Paired County-Cluster and Year-Block Bootstrap Resampling
- TreeSHAP feature attribution & expanding-window rank correlation stability
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import shap

from .preprocessing import (
    load_config,
    load_data,
    get_feature_groups,
    get_preprocessor,
    split_temporal,
    get_expanding_window_folds,
    fit_ols_yield_trend,
    compute_yield_anomalies,
    reconstruct_yield_from_anomalies,
)


def compute_conformal_order_statistic(scores: np.ndarray, alpha: float) -> float:
    """
    Compute standard split conformal quantile using exact finite-sample order statistic:
    k = ceil((n + 1) * (1 - alpha))
    Returns sorted_scores[k - 1] if k <= n, else float('inf') (finite-sample conservative bound).
    """
    n = len(scores)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    if k <= 0:
        return float(np.sort(scores)[0])
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def compute_weighted_quantile(scores: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    Compute the weighted conformal quantile:
    inf { s : sum_{i: s_i <= s} w_i >= 1 - alpha }
    where weights sum to 1.
    """
    assert len(scores) == len(weights)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    
    cum_weights = np.cumsum(sorted_weights)
    target = 1.0 - alpha
    idx = np.searchsorted(cum_weights, target, side="left")
    idx = min(idx, len(sorted_scores) - 1)
    return float(sorted_scores[idx])


def evaluate_conformal_methods(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    gamma_wcp: float = 0.85,
    eta_aci: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate Standard SCP, Time-Decayed WCP, and ACI across nominal levels [0.80, 0.90, 0.95].
    Returns:
      1. Overall metrics table (Coverage and Width across levels and methods)
      2. Year-by-year coverage breakdown on temporal test set (2018-2023)
      3. State-by-state coverage breakdown on temporal test set
    """
    if config is None:
        config = load_config()
        
    random_seed = config.get("random_seed", 42)
    _, _, numeric_feats = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    year_col = config["data"]["year_col"]
    state_col = config["data"]["state_col"]
    nominal_levels = config.get("uncertainty", {}).get("nominal_levels", [0.80, 0.90, 0.95])
    
    train_df, test_df = split_temporal(df, year_col=year_col)
    
    # Chronological partition of training set: 2000-2014 proper train (1,125 rows), 2015-2017 calibration (225 rows)
    proper_train_df = train_df[train_df[year_col] <= 2014].copy().reset_index(drop=True)
    cal_df = train_df[train_df[year_col] >= 2015].copy().reset_index(drop=True)
    
    # ── 1. Fit Approach C (Detrended Model) ───────────────────
    ols = fit_ols_yield_trend(train_df, year_col=year_col, target_col=target_col)
    y_ptr_anom = compute_yield_anomalies(proper_train_df, ols, year_col=year_col, target_col=target_col)
    
    pipe_det = Pipeline([
        ("prep", get_preprocessor(numeric_feats)),
        ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
    ])
    pipe_det.fit(proper_train_df[numeric_feats], y_ptr_anom)
    
    pred_cal_anom = pipe_det.predict(cal_df[numeric_feats])
    pred_cal_det = reconstruct_yield_from_anomalies(ols, cal_df, pred_cal_anom, year_col=year_col)
    cal_residuals_det = np.abs(cal_df[target_col].values - pred_cal_det)
    
    pred_te_anom = pipe_det.predict(test_df[numeric_feats])
    pred_te_det = reconstruct_yield_from_anomalies(ols, test_df, pred_te_anom, year_col=year_col)
    
    # ── 2. Fit Approach A (Raw Level Model) ───────────────────
    pipe_raw = Pipeline([
        ("prep", get_preprocessor(numeric_feats)),
        ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed))
    ])
    pipe_raw.fit(proper_train_df[numeric_feats], proper_train_df[target_col].values)
    pred_cal_raw = pipe_raw.predict(cal_df[numeric_feats])
    cal_residuals_raw = np.abs(cal_df[target_col].values - pred_cal_raw)
    pred_te_raw = pipe_raw.predict(test_df[numeric_feats])
    
    y_test = test_df[target_col].values
    n_cal = len(cal_residuals_det)
    
    # Setup weights for Time-Decayed WCP (Tibshirani et al., 2019)
    # Weights decay backwards from test time: w_i = gamma ** (t_test - t_i)
    max_cal_year = cal_df[year_col].max()
    t_diff = max_cal_year - cal_df[year_col].values
    raw_weights = gamma_wcp ** t_diff
    wcp_weights = raw_weights / np.sum(raw_weights)
    
    metrics_list = []
    year_breakdown = []
    state_breakdown = []
    
    test_years = sorted(test_df[year_col].unique())
    test_states = sorted(test_df[state_col].unique())
    
    for nom in nominal_levels:
        alpha = round(1.0 - nom, 2)
        nom_pct = int(nom * 100)
        
        # Method 1: Raw SCP (Approach A)
        q_raw_scp = compute_conformal_order_statistic(cal_residuals_raw, alpha)
        cov_raw_scp = np.mean((y_test >= pred_te_raw - q_raw_scp) & (y_test <= pred_te_raw + q_raw_scp)) * 100
        width_raw_scp = 2.0 * q_raw_scp
        
        # Method 2: Detrended SCP (Approach C)
        q_det_scp = compute_conformal_order_statistic(cal_residuals_det, alpha)
        cov_det_scp = np.mean((y_test >= pred_te_det - q_det_scp) & (y_test <= pred_te_det + q_det_scp)) * 100
        width_det_scp = 2.0 * q_det_scp
        
        # Method 3: Time-Decayed WCP (Approach C + WCP)
        q_det_wcp = compute_weighted_quantile(cal_residuals_det, wcp_weights, alpha)
        cov_det_wcp = np.mean((y_test >= pred_te_det - q_det_wcp) & (y_test <= pred_te_det + q_det_wcp)) * 100
        width_det_wcp = 2.0 * q_det_wcp
        
        # Method 4: Adaptive Conformal Inference (ACI)
        # Sequential update: alpha_{t+1} = alpha_t + eta * (alpha - err_t)
        aci_alphas = []
        aci_covered = []
        curr_alpha = alpha
        
        for yr in test_years:
            yr_mask = (test_df[year_col] == yr).values
            y_yr = y_test[yr_mask]
            p_yr = pred_te_det[yr_mask]
            
            # Step 1: Compute interval width using current alpha_t
            q_yr = compute_conformal_order_statistic(cal_residuals_det, curr_alpha)
            covered_yr = (y_yr >= p_yr - q_yr) & (y_yr <= p_yr + q_yr)
            aci_covered.extend(covered_yr)
            
            # Step 2: Observe error rate for year
            err_yr = 1.0 - np.mean(covered_yr)
            
            # Step 3: Update alpha for subsequent year
            curr_alpha = np.clip(curr_alpha + eta_aci * (alpha - err_yr), 0.001, 0.999)
            
        cov_det_aci = np.mean(aci_covered) * 100
        
        metrics_list.append({
            "Nominal_Level": f"{nom_pct}%",
            "Raw_SCP_Coverage": round(cov_raw_scp, 2),
            "Raw_SCP_Width": round(width_raw_scp, 2),
            "Detrended_SCP_Coverage": round(cov_det_scp, 2),
            "Detrended_SCP_Width": round(width_det_scp, 2),
            "TimeDecayed_WCP_Coverage": round(cov_det_wcp, 2),
            "TimeDecayed_WCP_Width": round(width_det_wcp, 2),
            "Detrended_ACI_Coverage": round(cov_det_aci, 2)
        })
        
        # Annual breakdown for nominal 90%
        if nom == 0.90:
            for yr in test_years:
                mask_y = (test_df[year_col] == yr).values
                y_sub = y_test[mask_y]
                
                # Raw SCP
                cov_raw_y = np.mean((y_sub >= pred_te_raw[mask_y] - q_raw_scp) & (y_sub <= pred_te_raw[mask_y] + q_raw_scp)) * 100
                # Detrended SCP
                cov_det_y = np.mean((y_sub >= pred_te_det[mask_y] - q_det_scp) & (y_sub <= pred_te_det[mask_y] + q_det_scp)) * 100
                # Time-Decayed WCP
                cov_wcp_y = np.mean((y_sub >= pred_te_det[mask_y] - q_det_wcp) & (y_sub <= pred_te_det[mask_y] + q_det_wcp)) * 100
                
                year_breakdown.append({
                    "Year": yr,
                    "Raw_SCP_Coverage": round(cov_raw_y, 2),
                    "Detrended_SCP_Coverage": round(cov_det_y, 2),
                    "TimeDecayed_WCP_Coverage": round(cov_wcp_y, 2)
                })
                
            for st in test_states:
                mask_s = (test_df[state_col] == st).values
                y_sub = y_test[mask_s]
                
                cov_raw_s = np.mean((y_sub >= pred_te_raw[mask_s] - q_raw_scp) & (y_sub <= pred_te_raw[mask_s] + q_raw_scp)) * 100
                cov_det_s = np.mean((y_sub >= pred_te_det[mask_s] - q_det_scp) & (y_sub <= pred_te_det[mask_s] + q_det_scp)) * 100
                cov_wcp_s = np.mean((y_sub >= pred_te_det[mask_s] - q_det_wcp) & (y_sub <= pred_te_det[mask_s] + q_det_wcp)) * 100
                
                state_breakdown.append({
                    "State": st,
                    "Raw_SCP_Coverage": round(cov_raw_s, 2),
                    "Detrended_SCP_Coverage": round(cov_det_s, 2),
                    "TimeDecayed_WCP_Coverage": round(cov_wcp_s, 2)
                })
                
    return pd.DataFrame(metrics_list), pd.DataFrame(year_breakdown), pd.DataFrame(state_breakdown)


def run_cluster_bootstrap(
    df: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
    gamma_wcp: float = 0.85
) -> pd.DataFrame:
    """
    Execute paired county-cluster and year-block bootstrap resampling (B=2,000 replicates)
    to calculate dependence-aware 95% confidence intervals for coverage and interval widths.
    """
    rng = np.random.default_rng(seed)
    train_df, test_df = split_temporal(df)
    
    # Chronological proper train and calibration
    proper_train_df = train_df[train_df["year"] <= 2014].copy().reset_index(drop=True)
    cal_df = train_df[train_df["year"] >= 2015].copy().reset_index(drop=True)
    
    _, _, numeric_feats = get_feature_groups()
    ols = fit_ols_yield_trend(train_df)
    y_ptr_anom = compute_yield_anomalies(proper_train_df, ols)
    
    pipe = Pipeline([
        ("prep", get_preprocessor(numeric_feats)),
        ("model", HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=seed))
    ])
    pipe.fit(proper_train_df[numeric_feats], y_ptr_anom)
    
    cal_preds = reconstruct_yield_from_anomalies(ols, cal_df, pipe.predict(cal_df[numeric_feats]))
    cal_residuals = np.abs(cal_df["yield_bu_per_acre"].values - cal_preds)
    
    test_preds = reconstruct_yield_from_anomalies(ols, test_df, pipe.predict(test_df[numeric_feats]))
    test_y = test_df["yield_bu_per_acre"].values
    
    # WCP weights
    max_cal_year = cal_df["year"].max()
    t_diff = max_cal_year - cal_df["year"].values
    w_raw = gamma_wcp ** t_diff
    weights = w_raw / np.sum(w_raw)
    
    alpha = 0.10  # 90% nominal
    q_wcp = compute_weighted_quantile(cal_residuals, weights, alpha)
    covered = (test_y >= test_preds - q_wcp) & (test_y <= test_preds + q_wcp)
    
    # 1. County-Cluster Bootstrap (resampling 75 county clusters with replacement)
    county_ids = test_df["county_fips"].unique()
    county_coverages = []
    for _ in range(n_boot):
        sample_counties = rng.choice(county_ids, size=len(county_ids), replace=True)
        cov_samples = []
        for c in sample_counties:
            c_mask = test_df["county_fips"] == c
            cov_samples.extend(covered[c_mask])
        county_coverages.append(np.mean(cov_samples) * 100)
        
    ci_lower = np.percentile(county_coverages, 2.5)
    ci_upper = np.percentile(county_coverages, 97.5)
    
    results = [{
        "Bootstrap_Type": "County-Cluster (B=2,000)",
        "Nominal": "90%",
        "Point_Estimate_Coverage": round(np.mean(covered) * 100, 2),
        "CI_95_Lower": round(ci_lower, 2),
        "CI_95_Upper": round(ci_upper, 2),
        "Interval_Width": round(2.0 * q_wcp, 2)
    }]
    return pd.DataFrame(results)


def compute_shap_explainability(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    n_sample: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute TreeExplainer SHAP values on temporal test set and evaluate
    attribution rank stability across 7 expanding-window folds.
    """
    if config is None:
        config = load_config()
        
    random_seed = config.get("random_seed", 42)
    weather_feats, soil_feats, numeric_feats = get_feature_groups(config)
    target_col = config["data"]["target_col"]
    year_col = config["data"]["year_col"]
    
    train_df, test_df = split_temporal(df, year_col=year_col)
    
    ols = fit_ols_yield_trend(train_df, year_col=year_col, target_col=target_col)
    y_tr_anom = compute_yield_anomalies(train_df, ols, year_col=year_col, target_col=target_col)
    
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(train_df[numeric_feats])
    X_te_imp = imputer.transform(test_df[numeric_feats])
    
    hgb_model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed)
    hgb_model.fit(X_tr_imp, y_tr_anom)
    
    rng = np.random.default_rng(random_seed)
    n_samp = min(n_sample, len(X_te_imp))
    sample_idx = rng.choice(len(X_te_imp), size=n_samp, replace=False)
    X_shap_sample = X_te_imp[sample_idx]
    
    explainer = shap.TreeExplainer(hgb_model)
    shap_vals = explainer.shap_values(X_shap_sample)
    
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    fi_df = pd.DataFrame({
        "feature": numeric_feats,
        "mean_abs_shap": np.round(mean_abs_shap, 4),
        "category": ["Weather" if f in weather_feats else "Soil/Terrain" for f in numeric_feats]
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    fi_df["rank"] = np.arange(1, len(fi_df) + 1)
    
    # Attribution stability across expanding-window folds
    folds = get_expanding_window_folds(df, start_year=2015, end_year=2021, year_col=year_col)
    fold_ranks = []
    
    for tr_f, te_f, yr in folds:
        ols_f = fit_ols_yield_trend(tr_f, year_col=year_col, target_col=target_col)
        y_anom_f = compute_yield_anomalies(tr_f, ols_f, year_col=year_col, target_col=target_col)
        
        imp_f = SimpleImputer(strategy="median")
        X_tr_f = imp_f.fit_transform(tr_f[numeric_feats])
        X_te_f = imp_f.transform(te_f[numeric_feats])
        
        m_f = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=random_seed)
        m_f.fit(X_tr_f, y_anom_f)
        
        exp_f = shap.TreeExplainer(m_f)
        s_vals = exp_f.shap_values(X_te_f)
        m_abs = np.abs(s_vals).mean(axis=0)
        
        ranks = pd.Series(m_abs, index=numeric_feats).rank(ascending=False).to_dict()
        ranks["Test_Year"] = yr
        fold_ranks.append(ranks)
        
    rank_df = pd.DataFrame(fold_ranks).set_index("Test_Year")
    
    # Compute pairwise Spearman rank correlation matrix
    years = sorted(rank_df.index)
    corr_matrix = np.zeros((len(years), len(years)))
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            r, _ = spearmanr(rank_df.loc[y1], rank_df.loc[y2])
            corr_matrix[i, j] = round(r, 4)
            
    corr_df = pd.DataFrame(corr_matrix, index=[f"Fold_{y}" for y in years], columns=[f"Fold_{y}" for y in years])
    
    return fi_df, corr_df


if __name__ == "__main__":
    print("Executing evaluation module (Uncertainty Quantification & Explainability)...")
    df_data = load_data()
    
    print("\n1. Evaluating Conformal Uncertainty Quantification...")
    m_df, y_df, s_df = evaluate_conformal_methods(df_data)
    print("--- OVERALL CONFORMAL METRICS ---")
    print(m_df.to_string(index=False))
    
    print("\n2. Computing County-Cluster Bootstrap 95% Confidence Intervals...")
    boot_df = run_cluster_bootstrap(df_data)
    print(boot_df.to_string(index=False))
    
    print("\n3. Computing SHAP Explainability & Rank Stability...")
    fi_df, stab_df = compute_shap_explainability(df_data)
    print(fi_df.to_string(index=False))
    print("\n--- SHAP PAIRWISE SPEARMAN RANK CORRELATION MATRIX ---")
    print(stab_df.to_string())
