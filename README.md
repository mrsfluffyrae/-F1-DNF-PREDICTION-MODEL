# 🏎️ F1 DNF Predictor

Which drivers are most likely to **not finish** Sunday's race?

This machine learning project predicts driver retirement probabilities using historical race results, constructor reliability trends, and starting grid positions. Rather than attempting to guess exact race incidents, the model outputs calibrated probabilities: *"Drivers in a scenario like this historically retire roughly X% of the time."*

Built around the **2026 Mexico City Grand Prix**, the pipeline can be adapted to any Formula 1 race once provided with a starting grid.

---

## 📊 Pipeline Overview

```text
[ Jolpica-F1 API ] 
        │
        ▼
01_fetch_data.py       ──► Downloads hybrid-era race results (2014–2026)
        │
        ▼
02_build_features.py   ──► Computes rolling reliability metrics (features.py)
        │
        ▼
03_train_model.py      ──► Trains Logistic Regression & HistGradientBoosting
        │
        ▼
04_predict_race.py     ──► Combines models with grid.csv to score race risks
```

---

## 🗃️ The Data

All data is collected via [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the open-source community successor to the Ergast API.

* **Coverage:** Complete race results from **2014 through 2026** (the turbo-hybrid era). Older eras were excluded because legacy mechanical failure rates distort modern reliability baselines.
* **Dataset Size:** ~5,100 driver-race entries (2014–2025) plus all completed rounds of the 2026 season.
* **DNF Definition:** A driver is classified as **Finished (`0`)** if their official classification is `Finished`, `Lapped`, or time/lap intervals (`+1 Lap`, `+2 Laps`, etc.). All other statuses (`Collision`, `Engine`, `Gearbox`, `Brakes`, `Retired`, etc.) are labeled as **DNF (`1`)**. Disqualified, excluded, or withdrawn entries are dropped.

---

## 🛠️ Feature Engineering

To prevent lookahead bias (data leakage), every rolling metric uses strict time-shifting (`.shift(1)`), ensuring features rely exclusively on information available **prior to lights out**:

| Feature | Description |
|---|---|
| `grid_pos` | Official starting grid position (pit lane starts default to 20th). |
| `pit_lane_start` | Binary flag (`1` if starting from the pit lane, `0` otherwise). |
| `driver_dnf_last10` | Driver retirement rate over their previous 10 starts. |
| `driver_dnf_career` | Cumulative career retirement rate prior to the current race. |
| `driver_prior_races` | Total career race starts (proxy for driver experience / rookie status). |
| `team_dnf_last10` | Constructor retirement rate over the previous 10 team starts. |
| `team_dnf_season_prior` | Constructor retirement rate within the current season. |
| `circuit_dnf_prior` | Historical retirement rate at the specific circuit. |

---

## 🤖 Model Architecture & Validation

* **Time-Based Split:** Trained on 2014–2022; validated out-of-time on 2023–2026. Random cross-validation was avoided to ensure the model never trains on future information.
* **Ensemble Strategy:** Blends a calibrated **Logistic Regression** model (linear interpretability) with a shallow **Histogram-Based Gradient Boosting Classifier** (`HistGradientBoostingClassifier`). The final risk score ($p_{\text{dnf}}$) is an unweighted blend of both.
* **Evaluation Metrics:** Brier score and Log Loss for probability calibration; PR-AUC and ROC-AUC for risk ranking. Standard classification accuracy is deliberately avoided due to target class imbalance (~14–18% base DNF rate).

### Benchmark Comparison

*Train samples: 3,689 (18.0% DNF) | Test samples: 1,782 (14.0% DNF)*

| Model | Log Loss ↓ | Brier Score ↓ | PR-AUC ↑ | ROC-AUC ↑ |
|---|---|---|---|---|
| **Baseline: Class Average** | 0.4103 | 0.1218 | 0.1397 | 0.5000 |
| **Baseline: Driver Last 10** | 0.4644 | 0.1275 | 0.1716 | 0.5430 |
| **HistGradientBoosting** | 0.4107 | 0.1221 | 0.1576 | 0.5394 |
| **Logistic Regression** | **0.3951** | **0.1178** | **0.2057** | **0.6200** |

### Probability Calibration

| Predicted Probability | Observed Retirement Rate |
|:---:|:---:|
| **10.0%** | 6.7% |
| **12.0%** | 10.1% |
| **14.0%** | 13.1% |
| **15.8%** | 16.2% |
| **17.7%** | 14.8% |
| **21.6%** | 22.9% |

### Historical High-Risk Out-of-Sample Predictions

| Season | Round | Grand Prix | Driver | Grid | Status | DNF | Predicted Risk |
|:---:|:---:|---|:---:|:---:|---|:---:|:---:|
| 2023 | 5 | Miami Grand Prix | SAR | 20 | Lapped | 0 | 31.1% |
| 2023 | 4 | Azerbaijan Grand Prix | GAS | 17 | Finished | 0 | 30.0% |
| 2026 | 15 | Azerbaijan Grand Prix | ALO | 21 | Retired | 1 | 29.7% |
| 2026 | 9 | British Grand Prix | ALO | 21 | Lapped | 0 | 29.6% |
| 2023 | 4 | Azerbaijan Grand Prix | OCO | 19 | Finished | 0 | 29.3% |
| 2023 | 4 | Azerbaijan Grand Prix | SAR | 14 | Finished | 0 | 29.2% |
| 2026 | 8 | Austrian Grand Prix | ALO | 21 | Lapped | 0 | 29.1% |
| 2026 | 17 | Singapore Grand Prix | ALO | 20 | Lapped | 0 | 28.4% |
| 2026 | 9 | British Grand Prix | STR | 22 | Lapped | 0 | 28.4% |
| 2026 | 3 | Japanese Grand Prix | STR | 22 | Retired | 1 | 27.8% |
| 2026 | 8 | Austrian Grand Prix | STR | 22 | Retired | 1 | 27.8% |
| 2026 | 3 | Japanese Grand Prix | ALO | 21 | Lapped | 0 | 27.4% |
| 2026 | 10 | Belgian Grand Prix | ALO | 22 | Lapped | 0 | 27.0% |
| 2026 | 15 | Azerbaijan Grand Prix | STR | 22 | Retired | 1 | 26.9% |
| 2026 | 6 | Monaco Grand Prix | STR | 22 | Retired | 1 | 26.8% |

---

## 🇲🇽 Case Study: 2026 Mexico City Grand Prix

Forecast generated prior to race start using the finalized post-penalty starting grid:

| Driver | Constructor | Grid | $p_{\text{dnf}}$ | $p_{\text{logistic}}$ | $p_{\text{boosting}}$ |
|---|---|:---:|:---:|:---:|:---:|
| **Lance Stroll** | Aston Martin | 20 | **37.8%** | 26.2% | 49.4% |
| **Sergio Pérez** | Cadillac | 21 | **36.9%** | 27.6% | 46.3% |
| **Valtteri Bottas** | Cadillac | 22 | **32.2%** | 27.1% | 37.2% |
| **Fernando Alonso** | Aston Martin | 18 | **26.0%** | 27.1% | 24.9% |
| **Carlos Sainz** | Williams | 16 | **19.9%** | 21.0% | 18.7% |

* **Expected Field Retirements:** ~3.2 cars (historical average: 3.7).
* **Lowest Risk Entries:** George Russell (6.3%), Lando Norris (8.2%), and Charles Leclerc (8.4%).

---

## 🛠️ Engineering Challenges & Bug Fixes

| Issue Encountered | Root Cause | Solution Implemented |
|---|---|---|
| `FastF1` authentication barrier | FastF1 now requires authenticated F1 TV credentials for live telemetry and lap timing streams. | Replaced FastF1 data ingestion with Jolpica-F1 REST endpoints, bypassing credentials for finishing classification. |
| Jolpica API `429 Too Many Requests` | Rapid, unthrottled per-race HTTP polling triggered server-side rate limits. | Implemented batch season fetching, exponential backoff, request sleep delays, and local disk caching. |
| Incomplete training records | Network drops caused silent row truncation down to 1,679 entries. | Added explicit response-size validations and cache integrity checks before writing out `f1_races_dataset.csv`. |
| Target label distortion | Slow cars classified as `Lapped` were improperly flagged as retirements. | Updated status filtering to explicitly treat `Lapped` and `+N Lap` results as race finishes. |
| Mid-season driver substitution | Stand-in driver (Hirakawa replacing Ocon) caused missing teammate references. | Added a `same_team_as` fallback parameter in `grid.csv` to inherit team reliability stats for debutants. |
| Data leakage across cars | Shared team reliability figures risked leaking current-race teammate outcomes. | Aggregated constructor reliability on a prior-race grouping level before merging across entries. |

---

## 🚀 Reproduction Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/f1-dnf-prediction.git
cd f1-dnf-prediction

# Create and activate an isolated virtual environment
python3 -m venv f1env
source f1env/bin/activate

# Install required dependencies
pip install pandas scikit-learn requests
```

### 2. Execution Pipeline

```bash
# 1. Fetch historical and current-season race records
python 01_fetch_data.py

# 2. Generate training features and outcome labels
python 02_build_features.py

# 3. Train models and benchmark out-of-time evaluation metrics
python 03_train_model.py

# 4. Predict DNF probabilities for the upcoming race
python 04_predict_race.py grid.csv
```

---

## 📁 Repository Structure

```text
├── 01_fetch_data.py          # Automated data collection client for Jolpica-F1
├── 02_build_features.py       # Pipeline creating time-lagged reliability features
├── 03_train_model.py          # Model training, calibration, and benchmark metrics
├── 04_predict_race.py         # Prediction scoring script for upcoming race grids
├── features.py                # Shared feature definitions and target label logic
├── grid.csv                   # Target race qualifying positions and driver mappings
├── mexico_dnf_predictions.csv # Scored output probabilities for the Mexico City GP
└── .gitignore                 # Excludes cache folders and local venvs
```

---

## ⚠️ Model Limitations & Future Work

* **Incident Unpredictability:** First-lap bottlenecks, variable track debris, and weather fluctuations carry an inherent degree of randomness that static pre-race data cannot fully anticipate.
* **Regulation Shifts:** The 2026 aerodynamic and power unit regulation reset weakens historical team reliability baselines from previous years.
* **Failure Decomposition:** Mechanical breakdowns and on-track collisions are currently aggregated into a single DNF class. Future iterations will treat these as distinct multi-class failure modes.

---

## 📜 Acknowledgments

* Race data provided by the [Jolpica-F1 project](https://github.com/jolpica/jolpica-f1).
* Exploration aided by [FastF1](https://github.com/theOehrly/Fast-F1).