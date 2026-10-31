"""
test_pipeline.py
----------------
Automated unit test suite verifying:
1. Dataset integrity, schema, and empirical provenance
2. Strict chronological isolation (zero temporal leakage)
3. Conformal prediction finite-sample quantile order statistics
4. Time-decayed weighted conformal quantile properties
5. In-sample detrending anomaly accuracy gains
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.preprocessing import (
    load_config,
    load_data,
    get_feature_groups,
    split_temporal,
    fit_ols_yield_trend,
    compute_yield_anomalies,
    reconstruct_yield_from_anomalies,
)
from src.model import run_temporal_shift_experiment
from src.evaluation import (
    compute_conformal_order_statistic,
    compute_weighted_quantile,
)


def test_dataset_integrity():
    """Verify dataset dimensions, absence of synthetic proxies, and complete empirical records."""
    df = load_data()
    
    # 1. Dimensions: 75 counties * 24 years = 1800 rows, 22 columns
    assert df.shape == (1800, 22), f"Expected (1800, 22), got {df.shape}"
    
    # 2. Missing & infinite values
    assert df.isnull().sum().sum() == 0, "Missing values found in empirical dataset!"
    assert not np.isinf(df.select_dtypes(include=np.number).values).any(), "Infinite values found!"
    
    # 3. Ensure synthetic proxy columns are completely removed
    synthetic_vars = ['soil_organic_matter', 'soil_ph', 'elevation_m', 'ndvi_peak', 'region']
    for v in synthetic_vars:
        assert v not in df.columns, f"Synthetic variable '{v}' found in dataset!"
        
    # 4. Ensure verified SSURGO and NOAA features are present
    empirical_vars = ['ssurgo_om_pct', 'ssurgo_ph', 'ssurgo_clay_pct', 'ssurgo_elev_m', 'yield_bu_per_acre', 'precip_july_mm']
    for v in empirical_vars:
        assert v in df.columns, f"Required empirical variable '{v}' missing!"
        
    # 5. Temporal bounds
    assert df['year'].min() == 2000 and df['year'].max() == 2023
    assert df['county_fips'].nunique() == 75
    assert df['state_name'].nunique() == 5


def test_chronological_isolation():
    """Verify that temporal splits have strictly zero forward leakage."""
    df = load_data()
    train_df, test_df = split_temporal(df, train_end_year=2017, test_start_year=2018)
    
    max_train_year = train_df['year'].max()
    min_test_year = test_df['year'].min()
    
    assert max_train_year == 2017
    assert min_test_year == 2018
    assert max_train_year < min_test_year, f"Leakage detected: train {max_train_year} >= test {min_test_year}"


def test_conformal_exact_order_statistic():
    """Verify exact finite-sample order statistic calculation: k = ceil((n + 1) * (1 - alpha))."""
    for n in [100, 270, 1000]:
        for alpha in [0.05, 0.10, 0.20]:
            k = int(np.ceil((n + 1) * (1 - alpha)))
            assert 1 <= k <= n
            
            mock_residuals = np.arange(1, n + 1, dtype=float)
            q_stat = compute_conformal_order_statistic(mock_residuals, alpha)
            assert q_stat == float(k), f"Expected order statistic {k}, got {q_stat}"
            
    # Finite-sample edge case verification: when k > n, q_hat must return inf
    small_residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # n = 5
    # For alpha = 0.10: k = ceil(6 * 0.90) = 6 > 5
    q_inf = compute_conformal_order_statistic(small_residuals, alpha=0.10)
    assert np.isinf(q_inf), f"Expected inf for k > n, got {q_inf}"


def test_weighted_conformal_properties():
    """Verify weighted quantile behavior and identity when weights are uniform."""
    scores = np.array([2.0, 5.0, 8.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    weights = np.ones_like(scores) / len(scores)
    
    q_w = compute_weighted_quantile(scores, weights, alpha=0.10)
    assert q_w == 35.0 or q_w == 40.0, f"Unexpected weighted quantile: {q_w}"


def test_detrending_statistical_advantage():
    """Verify detrended anomaly modeling cuts test MAE compared to raw baseline under temporal shift."""
    df = load_data()
    temp_df = run_temporal_shift_experiment(df)
    
    raw_hgb = temp_df[(temp_df['Approach'].str.contains('Raw')) & (temp_df['Model'] == 'HistGradientBoosting')].iloc[0]
    det_hgb = temp_df[(temp_df['Approach'].str.contains('Anomaly')) & (temp_df['Model'] == 'HistGradientBoosting')].iloc[0]
    
    assert det_hgb['Test_MAE'] < raw_hgb['Test_MAE'], "Detrended model failed to outperform raw model!"
    mae_diff = raw_hgb['Test_MAE'] - det_hgb['Test_MAE']
    pct_reduction = (mae_diff / raw_hgb['Test_MAE']) * 100
    assert pct_reduction > 50.0, f"Expected >50% MAE reduction, got {pct_reduction:.1f}%"


class TestCropYieldPipeline(unittest.TestCase):
    def test_dataset_integrity(self):
        test_dataset_integrity()

    def test_chronological_isolation(self):
        test_chronological_isolation()

    def test_conformal_exact_order_statistic(self):
        test_conformal_exact_order_statistic()

    def test_weighted_conformal_properties(self):
        test_weighted_conformal_properties()

    def test_detrending_statistical_advantage(self):
        test_detrending_statistical_advantage()


if __name__ == "__main__":
    unittest.main()
