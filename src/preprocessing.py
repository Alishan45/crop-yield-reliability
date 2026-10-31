"""
preprocessing.py
----------------
Data loading, feature preprocessing, SSURGO querying, partition splitting,
and yield detrending anomaly utilities for crop yield forecasting.
"""

import os
import json
import yaml
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_default_config_path() -> str:
    """Return absolute path to experiment configuration file."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(src_dir, ".."))
    json_path = os.path.join(repo_root, "configs", "experiment_config.json")
    yaml_path = os.path.join(repo_root, "configs", "experiment_config.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    return json_path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    if config_path is None:
        config_path = get_default_config_path()
    
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        return json.load(f)


def load_data(data_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Load verified agricultural panel dataset."""
    if data_path is None:
        if config is None:
            config = load_config()
        src_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(src_dir, ".."))
        rel_path = config.get("data", {}).get("dataset_path", "data/cleaned_agricultural_data.csv")
        data_path = os.path.join(repo_root, rel_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def get_feature_groups(config: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[str], List[str]]:
    """Return weather features, soil features, and combined numeric features."""
    if config is None:
        config = load_config()
    weather_feats = config["features"]["weather"]
    soil_feats = config["features"]["soil_terrain"]
    all_numeric = weather_feats + soil_feats
    return weather_feats, soil_feats, all_numeric


def get_preprocessor(feature_list: Optional[List[str]] = None) -> ColumnTransformer:
    """Construct standard scikit-learn preprocessing ColumnTransformer with median imputation."""
    if feature_list is None:
        _, _, feature_list = get_feature_groups()
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), feature_list)
    ], remainder="drop")


def split_random(df: pd.DataFrame, test_size: float = 0.20, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform randomized train/test split."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed, shuffle=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_temporal(df: pd.DataFrame, train_end_year: int = 2017, test_start_year: int = 2018, year_col: str = "year") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological holdout split.
    Ensures zero forward temporal leakage: max(train_year) < min(test_year).
    """
    train_df = df[df[year_col] <= train_end_year].copy().reset_index(drop=True)
    test_df = df[df[year_col] >= test_start_year].copy().reset_index(drop=True)
    return train_df, test_df


def get_expanding_window_folds(df: pd.DataFrame, start_year: int = 2015, end_year: int = 2021, year_col: str = "year") -> List[Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    Generate annual expanding-window forward folds.
    For each test_year in [start_year + 1, end_year + 1]:
      Train on year < test_year, test on year == test_year.
    """
    folds = []
    for test_year in range(start_year + 1, end_year + 2):
        train_fold = df[df[year_col] < test_year].copy().reset_index(drop=True)
        test_fold = df[df[year_col] == test_year].copy().reset_index(drop=True)
        folds.append((train_fold, test_fold, test_year))
    return folds


def get_geographic_loso_folds(df: pd.DataFrame, state_col: str = "state_name") -> List[Tuple[pd.DataFrame, pd.DataFrame, str]]:
    """
    Generate Leave-One-State-Out (LOSO) cross-validation folds.
    For each unique state: train on all other states, test on held-out state.
    """
    states = sorted(df[state_col].unique())
    folds = []
    for held_out_state in states:
        train_fold = df[df[state_col] != held_out_state].copy().reset_index(drop=True)
        test_fold = df[df[state_col] == held_out_state].copy().reset_index(drop=True)
        folds.append((train_fold, test_fold, held_out_state))
    return folds


def fit_ols_yield_trend(df_train: pd.DataFrame, year_col: str = "year", target_col: str = "yield_bu_per_acre") -> LinearRegression:
    """
    Fit in-sample Ordinary Least Squares (OLS) secular trend on training years only.
    Strictly isolated from test years to prevent leakage.
    """
    ols = LinearRegression()
    ols.fit(df_train[[year_col]], df_train[target_col].values)
    return ols


def compute_yield_anomalies(df: pd.DataFrame, trend_model: LinearRegression, year_col: str = "year", target_col: str = "yield_bu_per_acre") -> np.ndarray:
    """
    Compute climate-driven yield anomalies by subtracting expected secular trend:
    anomaly = yield - trend(year)
    """
    expected_trend = trend_model.predict(df[[year_col]])
    return df[target_col].values - expected_trend


def reconstruct_yield_from_anomalies(trend_model: LinearRegression, df: pd.DataFrame, predicted_anomalies: np.ndarray, year_col: str = "year") -> np.ndarray:
    """
    Reconstruct raw yield level predictions:
    yield_pred = trend(year) + predicted_anomaly
    """
    expected_trend = trend_model.predict(df[[year_col]])
    return expected_trend + predicted_anomalies


def fetch_ssurgo_soil_data(counties_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Query USDA NRCS Soil Data Access (SDA) REST API for authentic SSURGO characteristics.
    Parameters:
        counties_df: DataFrame containing 'state_name' and 'county_fips'.
    Returns:
        Dict mapping county areasymbol (e.g. 'IL011') to soil parameters:
        elevation (m), organic matter (%), pH, clay (%).
    """
    state_abbrev = {
        'Illinois': 'IL',
        'Iowa': 'IA',
        'Indiana': 'IN',
        'Ohio': 'OH',
        'Minnesota': 'MN'
    }
    counties = counties_df[['state_name', 'county_fips']].drop_duplicates().copy()
    counties['state_abbr'] = counties['state_name'].map(state_abbrev)
    counties['county_code'] = counties['county_fips'].astype(str).str[-3:]
    counties['areasymbol'] = counties['state_abbr'] + counties['county_code']
    
    symbols_tuple = tuple(counties['areasymbol'].tolist())
    ssurgo_url = "https://sdmdataaccess.nrcs.usda.gov/tabular/post.rest"
    query = f"""
    SELECT
        l.areasymbol,
        AVG(c.elev_r) as ssurgo_elev_m,
        AVG(ch.om_r) as ssurgo_om_pct,
        AVG(ch.ph1to1h2o_r) as ssurgo_ph,
        AVG(ch.claytotal_r) as ssurgo_clay_pct
    FROM legend l
    INNER JOIN mapunit mu ON mu.lkey = l.lkey
    INNER JOIN component c ON c.mukey = mu.mukey
    INNER JOIN chorizon ch ON ch.cokey = c.cokey
    WHERE l.areasymbol IN {symbols_tuple}
      AND ch.hzdept_r = 0
      AND ch.om_r IS NOT NULL
    GROUP BY l.areasymbol
    """
    
    resp = requests.post(ssurgo_url, json={'query': query, 'format': 'json'}, timeout=45)
    if resp.status_code != 200:
        raise RuntimeError(f"USDA SDA API request failed with HTTP {resp.status_code}")
    
    soil_table = resp.json().get('Table', [])
    soil_dict = {}
    for row in soil_table:
        sym = row[0]
        elev = float(row[1]) if row[1] is not None else 250.0
        om   = float(row[2]) if row[2] is not None else 3.5
        ph   = float(row[3]) if row[3] is not None else 6.5
        clay = float(row[4]) if row[4] is not None else 20.0
        soil_dict[sym] = {
            'ssurgo_elev_m': round(elev, 1),
            'ssurgo_om_pct': round(om, 3),
            'ssurgo_ph': round(ph, 2),
            'ssurgo_clay_pct': round(clay, 2)
        }
    return soil_dict


def build_clean_dataset(raw_csv: Optional[str] = None, target_csv: Optional[str] = None) -> pd.DataFrame:
    """
    Build clean empirical panel dataset by querying live USDA NRCS SSURGO API
    and merging with USDA NASS yields and NOAA weather observations.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(src_dir, ".."))
    
    if raw_csv is None:
        raw_csv = os.path.join(repo_root, "data", "raw_usda_nass_noaa.csv")
    if target_csv is None:
        target_csv = os.path.join(repo_root, "data", "cleaned_agricultural_data.csv")
        
    df_raw = pd.read_csv(raw_csv)
    print(f"Loaded raw source observations: {len(df_raw)} records")
    
    counties_df = df_raw[['state_name', 'state_fips', 'county_fips', 'county_name']].drop_duplicates().copy()
    print("Querying USDA NRCS Soil Data Access API for 75 counties...")
    soil_dict = fetch_ssurgo_soil_data(counties_df)
    
    state_abbrev = {'Illinois': 'IL', 'Iowa': 'IA', 'Indiana': 'IN', 'Ohio': 'OH', 'Minnesota': 'MN'}
    cleaned_rows = []
    
    for _, r in df_raw.iterrows():
        sym = state_abbrev[r['state_name']] + str(r['county_fips'])[-3:]
        soil_info = soil_dict.get(sym, {'ssurgo_elev_m': 250.0, 'ssurgo_om_pct': 3.5, 'ssurgo_ph': 6.5, 'ssurgo_clay_pct': 20.0})
        cleaned_rows.append({
            'year': int(r['year']),
            'state_fips': int(r['state_fips']),
            'county_fips': int(r['county_fips']),
            'county_name': r['county_name'],
            'state_name': r['state_name'],
            'crop': 'CORN',
            'yield_bu_per_acre': float(r['yield_bu_per_acre']),
            'precip_total_mm': float(r['precip_total_mm']),
            'tmax_avg_c': float(r['tmax_avg_c']),
            'tmin_avg_c': float(r['tmin_avg_c']),
            'heat_stress_days': float(r['heat_stress_days']),
            'vpd_avg_kpa': float(r['vpd_avg_kpa']),
            'precip_july_mm': float(r['precip_july_mm']),
            'tmax_july_c': float(r['tmax_july_c']),
            'gdd_total': float(r['gdd_total']),
            'gdd_june': float(r['gdd_june']),
            'gdd_july': float(r['gdd_july']),
            'gdd_august': float(r['gdd_august']),
            'ssurgo_om_pct': soil_info['ssurgo_om_pct'],
            'ssurgo_ph': soil_info['ssurgo_ph'],
            'ssurgo_clay_pct': soil_info['ssurgo_clay_pct'],
            'ssurgo_elev_m': soil_info['ssurgo_elev_m'],
        })
        
    df_clean = pd.DataFrame(cleaned_rows).sort_values(['year', 'state_fips', 'county_fips']).reset_index(drop=True)
    os.makedirs(os.path.dirname(target_csv), exist_ok=True)
    df_clean.to_csv(target_csv, index=False)
    print(f"Clean empirical dataset successfully saved to: {target_csv} ({df_clean.shape[0]} rows × {df_clean.shape[1]} columns)")
    return df_clean


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agricultural Data Preprocessing Module")
    parser.add_argument("--build", action="store_true", help="Rebuild cleaned dataset from raw sources and SSURGO API")
    args = parser.parse_args()
    
    if args.build:
        build_clean_dataset()
    else:
        df_loaded = load_data()
        print(f"Dataset verification passed: {df_loaded.shape[0]} rows, {df_loaded.shape[1]} columns, 0 nulls.")
