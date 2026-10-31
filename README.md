# Beyond Prediction Accuracy: Reliability and Explainability of Machine Learning Models for Crop Yield Forecasting Under Temporal and Geographic Shifts

[![Unit Tests](https://img.shields.io/badge/Unit%20Tests-PASS%20(5%2F5)-brightgreen)]()
[![Python 3.10 | 3.11](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue.svg)](https://zenodo.org/)
[![Data Provenance](https://img.shields.io/badge/Data%20Provenance-100%25%20Verified%20Empirical-success)]()

This repository provides the official, verified, and open-science research codebase for the study:  
**"Beyond Prediction Accuracy: Reliability and Explainability of Machine Learning Models for Crop Yield Forecasting Under Temporal and Geographic Shifts"**, targeted for publication in Elsevier's *Computers and Electronics in Agriculture* (COMPAG).

Author: **Ali Shan**  
Affiliation: Department of Computer Science, COMSATS University Islamabad, Wah Campus, Wah Cantt 47040, Pakistan.  
Correspondence: `alishan.cs01@gmail.com`

---

## 1. Overview and Key Contributions

Machine learning algorithms often achieve high cross-validation scores ($R^2 > 0.80$) under randomized train/test splits. In operational deployment, however, agricultural models face non-stationary secular technological drift and unmonitored geographies. This repository addresses these challenges with five core contributions:

1. **Failure Mode Analysis under Forward Temporal Drift:** Demonstrates that standard decision tree ensembles (Random Forest, HistGradientBoosting, LightGBM) experience catastrophic predictive failure ($R^2 = -1.30$, $\text{MAE} = 27.63\text{ bu/acre}$) when tested on future growing seasons (2018–2023) because partition algorithms cannot extrapolate linear secular yield trends (+2.09 bu/acre/year).
2. **In-Sample Yield Detrending (Approach C):** Decouples secular technological gains from high-frequency weather shocks, reducing out-of-sample temporal test MAE by **57.5%** down to **11.74 bu/acre** ($R^2 = +0.4713$, cluster bootstrap 95% CI: $[14.44, 17.36]$ bu/acre reduction, $p < 0.0005$).
3. **Conformal Uncertainty Quantification:** Identifies severe coverage collapse in Standard Split Conformal Prediction (SCP) under temporal drift (nominal 90% drops to 48.44% aggregate, and 17.33% in bumper crop seasons). Proves that **Time-Decayed Weighted Conformal Prediction (WCP, $\gamma=0.85$)** restores coverage to **88.44%** with narrow intervals (47.37 bu/acre).
4. **Geographic Generalization:** Evaluates spatial transferability via Leave-One-State-Out (LOSO) cross-validation across 5 major Corn Belt states, quantifying regional calibration deficits in Illinois and Ohio.
5. **Attribution Stability:** Evaluates TreeSHAP feature importances across sequential expanding folds, demonstrating a stable biophysical hierarchy ($r_s = 0.892, p < 0.001$) dominated by July precipitation, heat stress days, and vapor pressure deficit (VPD).

---

## 2. Quantitative Experimental Summary

| Metric / Evaluation Dimension | Raw Baseline (Approach A) | Proposed Detrended + WCP (Approach C) | Empirical Impact |
| :--- | :---: | :---: | :---: |
| **Temporal Test MAE (2018–2023)** | 27.63 bu/acre | **11.74 bu/acre** | **-57.5% Error Reduction** |
| **Temporal Test $R^2$** | -1.3007 (Collapse) | **+0.4713 (Strong Positive)** | **+1.772 $R^2$ Gain** |
| **Expanding Window Mean MAE** | 22.75 bu/acre | **12.44 bu/acre** | Consistent across 7 annual folds |
| **Geographic LOSO Mean MAE** | 22.40 bu/acre | **18.94 bu/acre** | All 5 states $R^2 > 0$ |
| **Conformal Coverage (Nominal 90%)** | 48.44% (Collapse) | **88.44% (Time-Decayed WCP)** | Deficit cut from 41.6% to 1.56% |
| **Conformal Interval Width** | 51.79 bu/acre | **47.37 bu/acre** | Tighter and distribution-free |
| **SHAP Rank Stability ($r_s$)** | Erratic on un-detrended | **Mean $r_s = 0.8922$ ($p < 0.001$)** | Exceptionally stable hierarchy |

---

## 3. Repository Directory Structure

```text
crop-yield-reliability/
├── README.md                          # Comprehensive documentation and reproduction guide
├── requirements.txt                   # Reproducible Python dependencies
├── LICENSE                            # MIT Open Source License
├── CITATION.cff                       # Citation File Format metadata for Zenodo & GitHub
├── configs/
│   ├── experiment_config.yaml         # Human-readable experiment hyperparameter specification
│   └── experiment_config.json         # Structured configuration mirror
├── data/
│   ├── cleaned_agricultural_data.csv  # Verified 1,800 county-years × 22 features (0 nulls)
│   ├── data_dictionary.csv            # Provenance, units, and column definitions
│   └── feature_provenance.csv         # Federal source mappings (USDA NASS, USDA NRCS, NOAA)
├── src/
│   ├── __init__.py                    # Module export definitions
│   ├── preprocessing.py               # Data loading, SSURGO querying, splitting & detrending
│   ├── model.py                       # Estimators, pipelines, Approach A/B/C, expanding/LOSO runners
│   └── evaluation.py                  # Conformal prediction (SCP/WCP/ACI), bootstrap CIs, TreeSHAP
├── notebooks/
│   └── experiments.ipynb              # Executed, publication-grade reproduction notebook
├── results/
│   ├── baseline_metrics.csv           # Random split (IID) performance
│   ├── temporal_metrics.csv           # Temporal shift comparison across Approaches A, B, C
│   ├── expanding_window_metrics.csv   # Annual rolling window metrics (2016–2022)
│   ├── geographic_metrics.csv         # Leave-One-State-Out (5 states) cross-validation
│   ├── uncertainty_metrics.csv        # Conformal coverage and interval widths (80%, 90%, 95%)
│   ├── uncertainty_by_year.csv        # Annual conformal coverage trajectories (2018–2023)
│   ├── uncertainty_by_state.csv       # State-by-state conformal coverage breakdowns
│   ├── conformal_bootstrap_results.csv# Paired county-cluster bootstrap 95% confidence intervals
│   ├── shap_feature_importance.csv    # Mean |SHAP| values and global feature ranks
│   ├── explanation_stability.csv      # Pairwise Spearman rank correlations across folds
│   ├── ablation_metrics.csv           # Feature group ablations (Weather, Soil, July critical)
│   └── figures/                       # Publication figures (High-resolution PNG)
└── tests/
    └── test_pipeline.py               # Automated test suite (data integrity, leakage, math proofs)
```

---

## 4. Installation and Setup

### Prerequisites
- Python 3.10 or 3.11
- Git

### Installation Steps
```bash
# 1. Clone repository
git clone https://github.com/alishan45/crop-yield-reliability.git
cd crop-yield-reliability

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. End-to-End Reproduction

### Step 1: Run Automated Verification Test Suite
Executes formal tests verifying zero temporal leakage, mathematical exactness of finite-sample conformal order statistics, weighted quantile properties, and detrending statistical significance:
```bash
python -m unittest tests/test_pipeline.py
```
*Expected Output: `Ran 5 tests ... OK`*

### Step 2: Run Baseline and Distribution Shift Experiments
Executes random split, temporal shift comparison (Approach A: Raw vs Approach B: Trend Feature vs Approach C: Detrended Anomaly), expanding window evaluation (2016–2022), geographic LOSO (5 states), and feature ablations:
```bash
python -m src.model
```

### Step 3: Run Conformal Uncertainty Quantification and SHAP Explainability
Computes Split Conformal Prediction (SCP), Time-Decayed Weighted Conformal Prediction (WCP, $\gamma=0.85$), Adaptive Conformal Inference (ACI), County-Cluster bootstrap intervals, and TreeSHAP rank stability matrices:
```bash
python -m src.evaluation
```

### Step 4: Interactive Visual Reproduction via Jupyter
Launch the pre-executed interactive notebook:
```bash
jupyter notebook notebooks/experiments.ipynb
```

---

## 6. Data Provenance & Open Science Policy

All data used in this study originate from public-domain United States federal government repositories:
- **Corn Yields:** USDA National Agricultural Statistics Service (NASS) Quick Stats API.
- **Climate Data:** NOAA National Centers for Environmental Information (NCEI) Climate at a Glance.
- **Soil Survey:** USDA Natural Resources Conservation Service (NRCS) SSURGO Database via Soil Data Access (SDA) REST API.

Detailed variable definitions, units, and citation provenance are cataloged in [`data/data_dictionary.csv`](data/data_dictionary.csv) and [`data/feature_provenance.csv`](data/feature_provenance.csv).

---

## 7. Citation

If you use this codebase, empirical dataset, or methodology in your research, please cite:

```bibtex
@article{shan2026cropreliability,
  title={Beyond Prediction Accuracy: Reliability and Explainability of Machine Learning Models for Crop Yield Forecasting Under Temporal and Geographic Shifts},
  author={Shan, Ali},
  journal={Computers and Electronics in Agriculture},
  year={2026},
  doi={10.5281/zenodo.XXXXXXX},
  url={https://github.com/alishan45/crop-yield-reliability}
}
```

---

## 8. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
