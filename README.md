# F1 DNF Predictor

Which drivers are most likely to *not finish* this Sunday's race?

This is a small machine learning project that tries to answer that question using free race results, and is upfront about how often it's wrong. (Spoiler: Formula 1 has opinions of its own, and some of them involve a Turn 1 collision.)

## What it does

For every driver on the grid, the model outputs a **probability of a DNF** (did not finish). It doesn't say "Stroll will crash". It says "drivers in a situation like this one retire roughly X% of the time". Think weather forecast, not crystal ball.

Built around the 2026 Mexico City Grand Prix, but it works for any race once you give it a starting grid.

## The data

Everything comes from [Jolpica-F1](https://github.com/jolpica/jolpica-f1), the community-run successor to the old Ergast API. It's free and needs no login. I pulled race results, grid positions and finishing status for **2014 to 2026**. That's the hybrid-engine era, since cars from the 1990s broke down far more often and would just confuse the model. That comes to just over 5,100 driver-race rows for 2014 to 2025, plus the 2026 season so far.

**How "DNF" is defined:** a driver finished if their status is `Finished`, `Lapped`, or `+1 Lap`, `+2 Laps` and so on. Everything else (collision, engine, gearbox, retired...) counts as a DNF. Disqualified, excluded and withdrawn entries are dropped, because those aren't real retirements.

## Features

Every feature only uses information that existed *before* the race started.

| Feature | What it captures |
|---|---|
| `grid_pos` | Starting position (pit lane start counts as 20th) |
| `pit_lane_start` | 1 if the driver starts from the pit lane |
| `driver_dnf_last10` | The driver's DNF rate over their previous 10 races |
| `driver_dnf_career` | The driver's DNF rate over all previous races |
| `driver_prior_races` | How many races they've started (a rough "rookie-ness") |
| `team_dnf_last10` | The team's DNF rate over its previous 10 races |
| `team_dnf_season_prior` | The team's DNF rate so far this season, which matters in a new-regulations year |
| `circuit_dnf_prior` | How chaotic this circuit has been in past years |

## How it was trained

- **Split by time, not at random.** Trained on 2014 to 2022, tested on 2023 onwards. A random split would let the model peek at the future, and real life doesn't allow that.
- **Two models:** logistic regression (simple, explainable) and gradient boosting (scikit-learn's `HistGradientBoostingClassifier`, kept shallow because there isn't much data). The final prediction is the average of the two.
- **Baselines to beat:** "everyone gets the average DNF rate", and "each driver's own recent DNF rate". If a fancy model can't beat those, it hasn't earned its keep.
- **Metrics:** log loss and Brier score (are the probabilities any good?), plus PR-AUC and ROC-AUC (does it rank risky drivers above safe ones?). Plain accuracy is useless here, because only about 15% of entries are DNFs and "everyone finishes" would score 85%.

### Results

**table printed by `python 03_train_model.py` .**

Train rows: 3689  |  Test rows: 1782
DNF rate  train: 0.180  test: 0.140

                   model  log_loss  brier  PR_AUC  ROC_AUC
  baseline: same for all    0.4103 0.1218  0.1397   0.5000
baseline: driver last 10    0.4644 0.1275  0.1716   0.5430
     logistic regression    0.3951 0.1178  0.2057   0.6200
       gradient boosting    0.4107 0.1221  0.1576   0.5394

How to read this: lower log_loss / brier = better probabilities; higher PR_AUC / ROC_AUC =better at ranking who is most at risk.

Calibration (logistic regression): when it says X%, how often do DNFs happen?
  predicted 10.0%  ->  actual  6.7%
  predicted 12.0%  ->  actual 10.1%
  predicted 14.0%  ->  actual 13.1%
  predicted 15.8%  ->  actual 16.2%
  predicted 17.7%  ->  actual 14.8%
  predicted 21.6%  ->  actual 22.9%

Top 15 highest-risk driver entries in the test period:
 season  round             race_name driver_code  grid   status  dnf  predicted_dnf_prob
   2023      5      Miami Grand Prix         SAR    20   Lapped    0               0.311
   2023      4 Azerbaijan Grand Prix         GAS    17 Finished    0               0.300
   2026     15 Azerbaijan Grand Prix         ALO    21  Retired    1               0.297
   2026      9    British Grand Prix         ALO    21   Lapped    0               0.296
   2023      4 Azerbaijan Grand Prix         OCO    19 Finished    0               0.293
   2023      4 Azerbaijan Grand Prix         SAR    14 Finished    0               0.292
   2026      8   Austrian Grand Prix         ALO    21   Lapped    0               0.291
   2026     17  Singapore Grand Prix         ALO    20   Lapped    0               0.284
   2026      9    British Grand Prix         STR    22   Lapped    0               0.284
   2026      3   Japanese Grand Prix         STR    22  Retired    1               0.278
   2026      8   Austrian Grand Prix         STR    22  Retired    1               0.278
   2026      3   Japanese Grand Prix         ALO    21   Lapped    0               0.274
   2026     10    Belgian Grand Prix         ALO    22   Lapped    0               0.270
   2026     15 Azerbaijan Grand Prix         STR    22  Retired    1               0.269
   2026      6     Monaco Grand Prix         STR    22  Retired    1               0.268

## Mexico City GP prediction

Made before the race, from the official starting grid after penalties. Top of the ranking:

| Driver | Grid | Predicted DNF probability |
|---|---|---|
| Stroll | 20 | 38% |
| Pérez | 21 | 37% |
| Bottas | 22 | 32% |
| Alonso | 18 | 26% |
| Sainz | 16 | 20% |

Lowest risk: Russell (6%), Norris and Leclerc (8% each). The model mostly likes front-row starters in reliable cars, which is not exactly a bold take.

**TODO: add what actually happened after the race.** One race proves very little either way, so treat it as a sanity check.

## Problems I ran into (and how they got fixed)

| Problem | Fix |
|---|---|
| FastF1 suddenly asked for an F1 account email, and an F1 TV subscription, on the first `session.load()` | The lap and telemetry data needs the subscription. Race results don't, so I dropped FastF1 and used Jolpica for results. |
| Jolpica replied `Too Many Requests` for about half of 2024 | The first script made one request per race, back to back. Now it fetches a whole season at a time, waits between requests, retries politely, and saves finished seasons to disk. |
| The first dataset was quietly incomplete (1,679 rows instead of a few thousand) | The same rate limit. Rows from the failed rounds were just missing, with no loud error. |
| Drivers with status `Lapped` were counted as DNFs | They finished, they were just slow. Found by reading the status counts instead of trusting my own code. |
| Risk of the model "seeing the future" through its features | Every rolling feature uses `.shift(1)` so the current race is excluded. Team stats are averaged per race first, so one car's result can't leak into its teammate's features. |
| Hirakawa replaced Ocon at Haas for this race, and the script took its line-up from the last race | The grid file now decides who's racing. A driver with no history goes in a teammate's team via the `same_team_as` column. |
| My first prediction had everyone on grid 10 | I'd saved the new scripts under a different name and was running the old ones. Check your files before panicking. |
| `SyntaxError: invalid non-printable character U+0016` | I pasted into the Python prompt with Ctrl+V. In a Linux terminal it's Ctrl+Shift+V. |

## Run it yourself

```bash
pip install pandas scikit-learn requests

python 01_fetch_data.py        # download results (a few minutes the first time)
python 02_build_features.py    # build the label and features
python 03_train_model.py       # train and test on a time-based split
python 04_predict_race.py grid.csv   # predict the next race
```

`grid.csv` has one row per driver:

```
driver_code,grid,same_team_as
HAM,1,
RUS,2,
...
HIR,19,BEA
```

Use `0` for a pit lane start. `same_team_as` is only needed for a driver with no race history.

To predict a different race, change the race settings at the top of `04_predict_race.py` (name, date, `CIRCUIT_ID`) and write a new grid file.

## What's in the repo

```
01_fetch_data.py      downloads results from Jolpica
features.py           the DNF label and feature code, shared by steps 2-4
02_build_features.py  builds the modelling table
03_train_model.py     trains, tests, compares against baselines
04_predict_race.py    ranks drivers by DNF probability for a race
grid.csv              the Mexico City GP starting grid
mexico_dnf_predictions.csv   the prediction itself
```

## Limitations

- **A lot of DNFs are just luck.** First-lap contact, debris and rain aren't in the data, so the model is better at "this car is unreliable" than "this driver will crash".
- **2026 is a new-regulations season.** Older history only partly applies, and there's little data on the newer teams and stand-in drivers (they get typical values filled in).
- **Collisions and mechanical failures are lumped together,** even though they have different causes.
- **No weather, strategy, penalties-in-race or engine-age data.** Some of the useful data needs a paid F1 TV subscription.
- **It's trained on a few thousand rows.** That's small, which is why the models are kept simple.

## Ideas for next time

- Split DNFs into mechanical vs incidents and model them separately.
- Add weather and safety-car history per circuit.
- Check how well calibrated the probabilities are across more races.

## Credits

Data from [Jolpica-F1](https://github.com/jolpica/jolpica-f1). Built with pandas and scikit-learn. I explored [FastF1](https://github.com/theOehrly/Fast-F1) before switching data sources. I built this with help from Claude (an AI assistant by Anthropic) for planning, code and debugging.