# 🏎️ F1 DNF Predictor

Which drivers are most likely to not finish Sunday's race?

This machine learning project estimates every driver's chance of a DNF (did not finish) using historical race results, team reliability trends, and starting grid positions. It doesn't try to guess exact incidents; it estimates baseline likelihoods (e.g., *"drivers in a scenario like this retire roughly X% of the time"*), operating more like a weather forecast than a crystal ball.

Built around the modern turbo-hybrid and current regulation eras, it can generate pre-race risk forecasts for any Grand Prix once supplied with a starting grid.

---

## 🏁 Track Record & Race Forecasts

| Grand Prix | Pre-Race Prediction | Post-Race Verification | Status |
| :--- | :--- | :--- | :--- |
| **Mexico City GP** | [`mexico_dnf_predictions.csv`](mexico_dnf_predictions.csv) | [`mexico_comparison.csv`](mexico_comparison.csv) | Completed |
| **São Paulo GP** | [`brazil_dnf_predictions.csv`](brazil_dnf_predictions.csv) | *Post-race verification* | Forecast Locked (Pre-Race Tagged) |

---

## 📊 Pipeline Overview

```text
[ Jolpica-F1 API ]
        │
        ▼
01_fetch_data.py        ──► Downloads race results (2014 to 2026)
        │
        ▼
02_build_features.py    ──► Builds the DNF label and rolling reliability features (via features.py)
        │
        ▼
03_train_model.py       ──► Trains Logistic Regression and Gradient Boosting, tests on later seasons
        │
        ▼
04_predict_race.py      ──► Scores the next race using grid.csv
        │
        ▼
05_compare_results.py   ──► After the race: prediction vs what actually happened
```

---

## 🗃️ The Data

All data comes from **Jolpica-F1**, the community-run successor to the Ergast API (free, open-access, no authentication required).

- **Coverage:** Every race from 2014 to the latest completed 2026 round. Older eras are excluded because pre-hybrid cars exhibited substantially higher mechanical attrition rates that distort modern baseline probabilities.
- **Size:** Over 5,100 driver-race rows spanning 2014–2025, plus current 2026 rounds.
- **DNF Definition:** A driver finished (`0`) if classified with status `Finished`, `Lapped`, `+1 Lap`, `+2 Laps`, etc. Non-finishes (`1`) encompass mechanical dropouts (`Engine`, `Gearbox`, `Brakes`, `Retired`) and track incidents (`Collision`, `Accident`). Disqualified, excluded, and withdrawn entries are omitted.

---

## 🛠️ Feature Engineering

Every rolling metric incorporates a strict `.shift(1)` offset to exclude the target race, ensuring zero data leakage before lights out.

| Feature | Description |
| :--- | :--- |
| `grid_pos` | Starting grid position (pit lane starts count as 20th / back of grid) |
| `pit_lane_start` | Binary flag (`1` if starting from the pit lane, `0` otherwise) |
| `driver_dnf_last10` | Driver's DNF frequency across their previous 10 starts |
| `driver_dnf_career` | Driver's career-long historical DNF rate |
| `driver_prior_races` | Total career race starts (experience proxy / rookie adjustment) |
| `team_dnf_last10` | Constructor's collective DNF frequency across the previous 10 events |
| `team_dnf_season_prior`| Constructor's DNF frequency within the current season to date |
| `circuit_dnf_prior` | Historical circuit attrition rate at this venue in prior editions |

---

## 🤖 Training and Validation

- **Temporal Validation Split:** Trained on 2014–2022 and evaluated out-of-sample on 2023–2026. Random K-fold splitting is deliberately avoided to prevent lookahead bias.
- **Model Architectures:**
  - **Logistic Regression:** Transparent, well-calibrated baseline for continuous risk estimation.
  - **Gradient Boosting (`HistGradientBoostingClassifier`):** Constrained shallow boosting suited for tabular datasets with limited row volume.
  - Final race forecasts blend both models using an unweighted ensemble average.
- **Evaluation Criteria:** Prioritizes **Log Loss** and **Brier Score** (probability calibration) alongside **PR-AUC** and **ROC-AUC** (relative risk discrimination). Raw classification accuracy is avoided due to severe class imbalance (~14–18% DNF base rate).

### Benchmark Comparison
*Train: 3,689 rows (18.0% DNF) | Test: 1,782 rows (14.0% DNF)*

| Model | Log Loss ↓ | Brier ↓ | PR-AUC ↑ | ROC-AUC ↑ |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline: same for all** | 0.4103 | 0.1218 | 0.1397 | 0.5000 |
| **Baseline: driver last 10** | 0.4644 | 0.1275 | 0.1716 | 0.5430 |
| **Gradient Boosting** | 0.4107 | 0.1221 | 0.1576 | 0.5394 |
| **Logistic Regression** | **0.3951** | **0.1178** | **0.2057** | **0.6200** |

*Takeaway:* Logistic regression delivers the strongest out-of-sample calibration and ranking power. Gradient boosting struggles to surpass simple naive baseline rates on log loss due to sample size constraints.

### Probability Calibration
Empirical evaluation of predicted probabilities vs. actual observed retirement frequencies:

| Metric | Bin 1 | Bin 2 | Bin 3 | Bin 4 | Bin 5 | Bin 6 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Predicted Risk** | 10.0% | 12.0% | 14.0% | 15.8% | 17.7% | 21.6% |
| **Observed Rate** | 6.7% | 10.1% | 13.1% | 16.2% | 14.8% | 22.9% |

### Highest-Risk Out-of-Sample Predictions
Top 15 riskiest predictions flagged by logistic regression in test evaluation (5 of 15 retired = 33.3%, vs. 29.0% expected):

| Season | Round | Grand Prix | Driver | Grid | Status | DNF | Predicted Risk |
| :---: | :---: | :--- | :---: | :---: | :--- | :---: | :---: |
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

## 🇧🇷 Forecast: 2026 São Paulo Grand Prix (2026-11-08)

Pre-race predictions generated prior to the start of the race at Interlagos:

| Driver | Constructor | Grid | P(DNF) | P(Logistic) | P(Boosting) | Driver DNF (Last 10) | Team DNF (Last 10) | Team DNF (Season Prior) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PER** | Cadillac | 22 | **0.31** | 0.31 | 0.31 | 0.30 | 0.45 | 0.39 |
| **BOT** | Cadillac | 18 | **0.29** | 0.25 | 0.33 | 0.60 | 0.45 | 0.39 |
| **STR** | Aston Martin | 17 | **0.25** | 0.27 | 0.24 | 0.70 | 0.50 | 0.50 |
| **ALO** | Aston Martin | 8 | **0.20** | 0.21 | 0.18 | 0.30 | 0.50 | 0.50 |
| **SAI** | Williams | 13 | **0.18** | 0.18 | 0.18 | 0.10 | 0.25 | 0.26 |
| **BOR** | Audi | 19 | **0.17** | 0.19 | 0.15 | 0.10 | 0.05 | 0.16 |
| **GAS** | Alpine | 16 | **0.16** | 0.18 | 0.15 | 0.10 | 0.20 | 0.13 |
| **COL** | Alpine | 15 | **0.16** | 0.18 | 0.14 | 0.30 | 0.20 | 0.13 |
| **VER** | Red Bull | 21 | **0.16** | 0.18 | 0.13 | 0.10 | 0.05 | 0.16 |
| **LEC** | Ferrari | 14 | **0.15** | 0.15 | 0.16 | 0.10 | 0.10 | 0.11 |
| **HAM** | Ferrari | 20 | **0.15** | 0.18 | 0.12 | 0.10 | 0.10 | 0.11 |
| **LIN** | RB | 12 | **0.14** | 0.14 | 0.15 | 0.10 | 0.05 | 0.08 |
| **ALB** | Williams | 9 | **0.13** | 0.15 | 0.12 | 0.40 | 0.25 | 0.26 |
| **PIA** | McLaren | 10 | **0.13** | 0.14 | 0.13 | 0.10 | 0.10 | 0.18 |
| **CAM** | Haas | 11 | **0.12** | 0.17 | 0.07 | NaN | 0.20 | 0.18 |
| **BEA** | Haas | 6 | **0.10** | 0.13 | 0.07 | 0.30 | 0.20 | 0.18 |
| **HAD** | Red Bull | 5 | **0.10** | 0.11 | 0.09 | 0.00 | 0.05 | 0.16 |
| **RUS** | Mercedes | 7 | **0.09** | 0.11 | 0.08 | 0.20 | 0.10 | 0.11 |
| **LAW** | RB | 3 | **0.09** | 0.10 | 0.08 | 0.00 | 0.05 | 0.08 |
| **HUL** | Audi | 4 | **0.09** | 0.10 | 0.07 | 0.00 | 0.10 | 0.16 |
| **ANT** | Mercedes | 2 | **0.08** | 0.10 | 0.07 | 0.00 | 0.10 | 0.11 |
| **NOR** | McLaren | 1 | **0.07** | 0.09 | 0.04 | 0.10 | 0.10 | 0.18 |

- **Expected number of DNFs this race (sum of probabilities):** `3.3`
- **Historical field baseline:** `3.7`

---

## 🇲🇽 Case Study: 2026 Mexico City Grand Prix

The forecast was generated prior to lights out from the entered starting grid:

| Driver | Team | Grid Used | Ensemble Blend | Logistic | Boosting |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Stroll** | Aston Martin | 20 | **37.8%** | 26.2% | 49.4% |
| **Pérez** | Cadillac | 21 | **36.9%** | 27.6% | 46.3% |
| **Bottas** | Cadillac | 22 | **32.2%** | 27.1% | 37.2% |
| **Alonso** | Aston Martin | 18 | **26.0%** | 27.1% | 24.9% |
| **Sainz** | Williams | 16 | **19.9%** | 21.0% | 18.7% |

- **Expected retirements:** ~3.2 cars (historical average: 3.7).
- **Lowest predicted risk:** Russell (6.3%), Norris (8.2%), Leclerc (8.4%).

### Actual Race Outcome

Verstappen won, Hamilton placed second, and Antonelli finished third. Five drivers retired: Stroll (technical), Alonso (technical), Bottas (crash), Bearman (damage after restart contact), and Lindblad (collision).

| Risk Rank (of 22) | Driver | Predicted Risk | Actual Outcome |
| :---: | :--- | :---: | :--- |
| **1** | Stroll | 37.8% | **DNF (technical)** |
| **2** | Pérez | 36.9% | Finished (P17) |
| **3** | Bottas | 32.2% | **DNF (crash)** |
| **4** | Alonso | 26.0% | **DNF (technical)** |
| **5** | Sainz | 19.9% | Finished (P15) |
| **6** | Bearman | ~16.0% | **DNF (damage)** |
| **16** | Lindblad | ~10.0% | **DNF (collision)** |

- **3 of the model's top 4 riskiest drivers retired**, as did **4 of its top 6**.
- The model projected 3.2 retirements vs. 5 actual DNFs, underestimating late-race incident density.
- Full per-driver breakdown is preserved in [`mexico_comparison.csv`](mexico_comparison.csv).

> **A mistake worth owning:** The grid fed into the model included a manual entry error: Liam Lawson started from the pit lane due to a qualifying penalty but was input as 9th, shifting grid spots down by one for cars behind him. The pre-race forecast was preserved without revision to maintain integrity.

---

## 🧩 Debugging & Pipeline Challenges

| Problem | Root Cause | Fix Applied |
| :--- | :--- | :--- |
| **FastF1 authentication requirement** | FastF1 API began requiring credentials for high-tier telemetry | Switched ingest pipeline to the open **Jolpica-F1** API. |
| **Jolpica `HTTP 429: Too Many Requests`** | Pipeline sent sequential single-race calls too rapidly | Switched to full-season batch queries, implemented exponential backoff, and added local caching. |
| **Missing dataset rows (~1,679 rows initially)** | Silent failures on rate-limited endpoints caused missing rounds | Implemented strict completeness checks ensuring each season fetches expected rounds before saving. |
| **Lapped cars categorized as DNFs** | Rule logic misclassified non-lead-lap finishers as non-classified | Refined parser: `Lapped`, `+1 Lap`, `+2 Laps`, etc., evaluate strictly as finished (`0`). |
| **Data leakage (lookahead bias)** | Feature extraction used aggregated season statistics containing future outcomes | Enforced `.shift(1)` across all rolling metrics and computed constructor stats at the race level before assigning to drivers. |
| **Driver substitute edge cases** | Roster changes (e.g., reserve drivers) broke static driver lookups | Ingested `grid.csv` mapping and assigned unknown rookie entries a teammate baseline via `same_team_as`. |
| **Overwritten local predictions** | Naming collisions in iterative prediction runs | Standardized explicit race-scoped output filenames (`grid_brazil.csv`, `brazil_dnf_predictions.csv`). |

---

## 🚀 Run It Yourself

```bash
# Clone the repository
git clone https://github.com/mrsfluffyrae/-F1-DNF-PREDICTION-MODEL.git
cd -F1-DNF-PREDICTION-MODEL

# Setup virtual environment
python3 -m venv f1env
source f1env/bin/activate
pip install pandas scikit-learn requests matplotlib

# Execute the pipeline
python 01_fetch_data.py              # Download historical results via Jolpica
python 02_build_features.py          # Compile features and DNF target labels
python 03_train_model.py             # Train models & evaluate baseline metrics
python 04_predict_race.py grid.csv   # Generate pre-race DNF risk predictions
python 05_compare_results.py         # Post-race verification and metric logging
```

### `grid.csv` Format

```csv
driver_code,grid,same_team_as
HAM,1,
RUS,2,
...
HIR,19,BEA
```
*Use `0` or back-of-grid index for pit lane starts. `same_team_as` is used for reserve or new drivers lacking historical data.*

---

## 📁 Repository Structure

```text
├── .gitignore                    # Local environment, cache, and virtualenv ignores
├── 01_fetch_data.py              # Ingests race data from Jolpica-F1 API
├── features.py                   # Feature engineering logic shared across modules
├── 02_build_features.py          # Compiles modeling dataset
├── 03_train_model.py             # Trains models and generates benchmark metrics
├── 04_predict_race.py            # Generates pre-race DNF probabilities for an upcoming GP
├── 05_compare_results.py         # Evaluates predictions against official race classifications
├── grid_template.csv             # Template for entering race grid positions
├── grid_brazil.csv               # Input grid configuration for the São Paulo GP
├── brazil_dnf_predictions.csv    # Official model forecast output for São Paulo
├── mexico_dnf_predictions.csv    # Official model forecast output for Mexico City
└── mexico_comparison.csv         # Post-race performance verification for Mexico City
```

---

## ⚠️ Limitations & Next Steps

- **Inherent Stochasticity:** First-lap contacts, unpredictable weather swings, and puncture-causing debris have weak statistical correlation with historical rolling reliability.
- **New Regulation Cycles:** Multi-year baseline metrics face structural degradation during major engine or aero regulation shifts.
- **Model Blending:** The pipeline currently uses an unweighted average of Logistic Regression and HistGradientBoosting; tuning weights based on out-of-sample log loss is planned.
- **Failure Mode Separation:** Splitting mechanical dropouts from driver accidents into separate multi-class or distinct survival models.

---

## 📜 Acknowledgments

- Race telemetry and historical classifications provided by the **Jolpica-F1** project.
- Built using **pandas**, **scikit-learn**, and Python.