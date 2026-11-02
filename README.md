# 🏎️ F1 DNF Predictor

Which drivers are most likely to **not finish** Sunday's race?

This machine learning project estimates every driver's chance of a DNF (did not finish) using historical race results, team reliability trends and starting grid positions. It doesn't try to guess exact incidents. It says things like *"drivers in a spot like this retire roughly X% of the time"*, which is closer to a weather forecast than a crystal ball.

Built around the **2026 Mexico City Grand Prix**, but it can be used for any race once you give it a starting grid.

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

All data comes from [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the community-run successor to the Ergast API. It's free and needs no login.

* **Coverage:** every race from **2014 to the latest completed 2026 round** (the turbo-hybrid era). Older cars broke down far more often, which would muddy the picture for modern cars.
* **Size:** just over 5,100 driver-race rows for 2014 to 2025, plus the 2026 season so far.
* **DNF definition:** a driver **finished (`0`)** if their status is `Finished`, `Lapped`, or `+1 Lap`, `+2 Laps` and so on. Anything else (`Collision`, `Engine`, `Gearbox`, `Brakes`, `Retired`...) is a **DNF (`1`)**. Disqualified, excluded and withdrawn entries are dropped.

---

## 🛠️ Features

Every rolling metric uses `.shift(1)`, which skips the current race, so features only contain information available **before lights out**.

| Feature | Description |
|---|---|
| `grid_pos` | Starting grid position (a pit lane start counts as 20th) |
| `pit_lane_start` | `1` if starting from the pit lane, otherwise `0` |
| `driver_dnf_last10` | Driver's DNF rate over their previous 10 races |
| `driver_dnf_career` | Driver's DNF rate over all previous races |
| `driver_prior_races` | Career starts so far (a rough proxy for experience / rookie status) |
| `team_dnf_last10` | Team's DNF rate over its previous 10 races |
| `team_dnf_season_prior` | Team's DNF rate so far this season |
| `circuit_dnf_prior` | How often entries have retired at this circuit in past years |

---

## 🤖 Training and Validation

* **Split by time, not at random:** trained on 2014 to 2022, tested on 2023 onwards (2026 included). A random split would let the model peek at the future.
* **Two models:** logistic regression (simple, explainable) and gradient boosting (`HistGradientBoostingClassifier`, kept shallow because there isn't much data). The forecast for a race uses the **unweighted average of both**.
* **Metrics:** log loss and Brier score (are the probabilities any good?), plus PR-AUC and ROC-AUC (does it rank risky drivers above safe ones?). Plain accuracy is deliberately avoided: with a 14 to 18% DNF rate, "everybody finishes" already scores 82 to 86%.

### Benchmark Comparison

*Train: 3,689 rows (18.0% DNF) | Test: 1,782 rows (14.0% DNF)*

| Model | Log Loss ↓ | Brier ↓ | PR-AUC ↑ | ROC-AUC ↑ |
|---|---|---|---|---|
| Baseline: same for all | 0.4103 | 0.1218 | 0.1397 | 0.5000 |
| Baseline: driver last 10 | 0.4644 | 0.1275 | 0.1716 | 0.5430 |
| Gradient boosting | 0.4107 | 0.1221 | 0.1576 | 0.5394 |
| **Logistic regression** | **0.3951** | **0.1178** | **0.2057** | **0.6200** |

What that says, honestly: **logistic regression is the only model that clearly beats both baselines**, and the gain is modest but real (a random ranking would score a PR-AUC of about 0.14, the DNF rate). Gradient boosting did not beat "just guess the average" on log loss, which is typical when a flexible model is trained on a few thousand rows.

### Probability Calibration

When the logistic regression says X%, how often do drivers really retire?

| Predicted | 10.0% | 12.0% | 14.0% | 15.8% | 17.7% | 21.6% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Observed** | 6.7% | 10.1% | 13.1% | 16.2% | 14.8% | 22.9% |

### Highest-Risk Out-of-Sample Predictions

The 15 entries the logistic regression rated riskiest in the test period. **5 of the 15 retired (33%)**, against about 29% predicted on average.

| Season | Round | Grand Prix | Driver | Grid | Status | DNF | Predicted risk |
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

The forecast was generated before the race from the starting grid as I entered it (see the grid note below). Top of the ranking:

| Driver | Team | Grid used | Blend | Logistic | Boosting |
|---|---|:---:|:---:|:---:|:---:|
| Stroll | Aston Martin | 20 | **37.8%** | 26.2% | 49.4% |
| Pérez | Cadillac | 21 | **36.9%** | 27.6% | 46.3% |
| Bottas | Cadillac | 22 | **32.2%** | 27.1% | 37.2% |
| Alonso | Aston Martin | 18 | **26.0%** | 27.1% | 24.9% |
| Sainz | Williams | 16 | **19.9%** | 21.0% | 18.7% |

* **Expected retirements:** about 3.2 cars (historical average: 3.7).
* **Lowest risk:** Russell (6.3%), Norris (8.2%) and Leclerc (8.4%).

### What actually happened

Verstappen won, Hamilton was second and Antonelli's third place clinched the championship. **Five drivers retired**: Stroll (technical), Alonso (technical), Bottas (crash), Bearman (damage after contact on the restart) and Lindblad (taken out by Piastri).

| Risk rank (of 22) | Driver | Predicted | What happened |
|:---:|---|:---:|---|
| 1 | Stroll | 37.8% | DNF (technical) |
| 2 | Pérez | 36.9% | Finished (P17) |
| 3 | Bottas | 32.2% | DNF (crash) |
| 4 | Alonso | 26.0% | DNF (technical) |
| 5 | Sainz | 19.9% | Finished (P15) |
| 6 | Bearman | about 16% | DNF (damage) |
| 16 | Lindblad | about 10% | DNF (collision) |

* **3 of the model's 4 riskiest drivers retired, and 4 of its top 6.** Pérez was a miss.
* **5 drivers retired against 3.2 expected,** so the model underestimated how chaotic the race was.
* Lindblad's retirement was a collision caused by another driver, which no pre-race data can predict.
* Full per-driver output is in `mexico_comparison.csv`, produced by `05_compare_results.py`.

![Predicted risk vs what happened](mexico_dnf_chart.png)

One race proves very little either way. Treat this as a sanity check, not proof.

### A mistake worth owning

The grid I fed the model had a data-entry error. Liam Lawson was disqualified from qualifying and started from the **pit lane**, but I entered him in 9th. That pushed every driver from Gasly down to Bottas back by one place (Stroll was entered 20th but started 19th, for example). The model only sees grid position as a single number, so the effect on the forecast is probably small, but the inputs weren't what they should have been. I kept the original forecast as it was instead of re-running it after the race, because that would no longer be a prediction.

---

## 🧩 Problems I Ran Into

| Problem | What was going on | Fix |
|---|---|---|
| `FastF1` asked for an F1 account email | FastF1 now needs an F1 TV subscription for most of its data | Dropped it for ingestion. Finishing status comes from Jolpica, which needs no login. |
| Jolpica `429 Too Many Requests` | The first script fired one request per race, back to back | Fetch a whole season per request, wait between requests, retry with longer pauses, and save finished seasons to disk |
| Only 1,679 rows in the first dataset | Same rate limit. Rounds that failed simply vanished, with no loud error | Seasons are fetched in full and saved only when complete, and the script prints races per season (about 16 to 24 is right) |
| `Lapped` drivers counted as DNFs | They finished, just slowly. The label code only recognised `+1 Lap` style statuses | Found by reading the status counts. `Lapped` now counts as finished. |
| Risk of the model seeing the future | Features built from results could include the race being predicted | Rolling features use `.shift(1)`. Team stats are averaged per race first, so one car's result can't leak into its teammate's features. |
| Hirakawa replaced Ocon at Haas | The script took its line-up from the last race, so it still listed Ocon | The grid file now decides who races, and a driver with no history inherits a teammate's team via `same_team_as` |
| First prediction put everyone on grid 10 | New scripts were saved under different names, so the old ones ran | Check your files before panicking |
| Wrong starting grid for Mexico | Lawson's pit lane start was missed, shifting 13 drivers by one place | Disclosed above. Next time: check the grid against the official classification before running. |

---

## 🚀 Run It Yourself

```bash
git clone https://github.com/mrsfluffyrae/f1-dnf-prediction.git
cd f1-dnf-prediction

python3 -m venv f1env
source f1env/bin/activate
pip install pandas scikit-learn requests matplotlib

python 01_fetch_data.py              # download results (a few minutes the first time)
python 02_build_features.py          # build the label and features
python 03_train_model.py             # train and test on a time-based split
python 04_predict_race.py grid.csv   # predict the next race
python 05_compare_results.py         # after the race: compare with the real result
```

`grid.csv` has one row per driver:

```text
driver_code,grid,same_team_as
HAM,1,
RUS,2,
...
HIR,19,BEA
```

Use `0` for a pit lane start. `same_team_as` is only needed for a driver with no race history. To predict a different race, change the race settings at the top of `04_predict_race.py` and `05_compare_results.py` (name, date, circuit) and write a new grid file.

---

## 📁 What's in the Repo

```text
01_fetch_data.py             downloads results from Jolpica
features.py                  the DNF label and feature code, shared by steps 2 to 4
02_build_features.py         builds the modelling table
03_train_model.py            trains, tests, compares against baselines
04_predict_race.py           ranks drivers by DNF probability for a race
05_compare_results.py        compares the prediction with the real result
grid.csv                     the Mexico City GP grid exactly as it was fed to the model
mexico_dnf_predictions.csv   the forecast itself
mexico_comparison.csv        forecast vs actual outcome, per driver
mexico_dnf_chart.png         chart used in this README
.gitignore                   keeps caches and virtual environments out of git
```

---

## ⚠️ Limitations and What I'd Do Next

* **A lot of DNFs are luck.** First-lap contact, debris and weather aren't in the data, so the model is better at "this car is unreliable" than "this driver will crash".
* **2026 is a new-regulations season.** Older history only partly applies, and newer teams and stand-in drivers have little data (they get typical values filled in).
* **The forecast blends two models that I only tested separately.** Given the benchmark, the next step is to evaluate the blend, or use logistic regression alone.
* **Collisions and mechanical failures are lumped together,** even though they have different causes. Splitting them into two models is the obvious upgrade.
* **Useful data sits behind a paywall.** Lap, tyre and telemetry data would need an F1 TV subscription.

---

## 📜 Acknowledgments

* Race data from the [Jolpica-F1 project](https://github.com/jolpica/jolpica-f1).
* Built with pandas and scikit-learn. [FastF1](https://github.com/theOehrly/Fast-F1) was explored before switching data sources.
