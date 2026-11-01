"""
STEP 4: Predict who is most likely to DNF in the upcoming race.

Run it two ways:
    python 04_predict_race.py              (before qualifying: no grid yet)
    python 04_predict_race.py grid.csv     (after Saturday's qualifying)

The first run also writes grid_template.csv listing the driver codes, so you
only have to fill in the starting positions.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, add_features, add_label

# ----------------------------------------------------------- race settings
SEASON = 2026
RACE_NAME = "Mexico City Grand Prix"
RACE_DATE = "2026-11-01"
CIRCUIT_ID = "rodriguez"     # how the data names Autodromo Hermanos Rodriguez
DEFAULT_GRID = 10            # placeholder used for everyone until qualifying is done

# ------------------------------------------------------------ load history
raw = pd.read_csv("f1_races_dataset.csv")
history = add_label(raw)
history["is_future"] = False

last = raw.sort_values(["season", "round"]).iloc[-1]
last_season, last_round = int(last["season"]), int(last["round"])
print(f"Latest race in your data: {last['race_name']} ({last_season}, round {last_round})")
if last_season < SEASON:
    print(f"WARNING: no {SEASON} races found. Re-run 01_fetch_data.py. The line-up below "
          f"would be last year's, which is wrong.")
if CIRCUIT_ID not in set(raw["circuit_id"]):
    print(f"WARNING: circuit '{CIRCUIT_ID}' not found in the data; the circuit feature will be empty.")

# ------------------------- who is racing, and where they start
if len(sys.argv) > 1:
    # The grid file DEFINES who is racing (so late driver changes are handled).
    grid = pd.read_csv(sys.argv[1])
    grid["driver_code"] = grid["driver_code"].astype(str).str.strip().str.upper()

    # For drivers already in your data, look up their ids from their latest race.
    latest = (raw.sort_values(["season", "round"])
                 .drop_duplicates("driver_code", keep="last")
                 [["driver_code", "driver_id", "constructor_id"]])
    roster = grid.merge(latest, on="driver_code", how="left")
    if "same_team_as" not in roster.columns:
        roster["same_team_as"] = np.nan

    # Drivers with no race history (e.g. a stand-in) are placed in a teammate's team.
    team_of = dict(zip(roster["driver_code"], roster["constructor_id"]))
    for i in roster[roster["driver_id"].isna()].index:
        code = roster.at[i, "driver_code"]
        mate = roster.at[i, "same_team_as"]
        team = team_of.get(str(mate).strip().upper()) if pd.notna(mate) else None
        if team is None or pd.isna(team):
            sys.exit(f"ERROR: '{code}' has no race history. In the grid file, put a teammate's "
                     f"driver code in the same_team_as column for that row.")
        roster.at[i, "driver_id"] = "new_" + code.lower()
        roster.at[i, "constructor_id"] = team
        print(f"NOTE: {code} has no race history; treated as a new driver in the same team as {mate}.")

    roster["grid"] = roster["grid"].astype(int)
    print(f"Using starting grid from {sys.argv[1]} ({len(roster)} drivers)")
    if roster["grid"].duplicated().any():
        print("WARNING: two drivers share a grid position. Check the file.")
else:
    roster = (raw[(raw["season"] == last_season) & (raw["round"] == last_round)]
              [["driver_id", "driver_code", "constructor_id"]]
              .drop_duplicates("driver_id").reset_index(drop=True))
    roster["grid"] = DEFAULT_GRID
    roster[["driver_code"]].assign(grid="", same_team_as="").to_csv("grid_template.csv", index=False)
    print("No grid file given: everyone gets the same placeholder grid position, so this run "
          "only reflects reliability history. Fill in 'grid_template.csv' after qualifying and "
          "re-run with it.")

# ------------------------------------- add the upcoming race as "future" rows
future = roster[["driver_id", "driver_code", "constructor_id", "grid"]].copy()
future["season"] = SEASON
future["round"] = last_round + 1 if last_season == SEASON else 1
future["race_name"] = RACE_NAME
future["circuit_id"] = CIRCUIT_ID
future["date"] = RACE_DATE
future["dnf"] = np.nan          # unknown: that's what we're predicting
future["is_future"] = True

combined = pd.concat([history, future], ignore_index=True)
feat = add_features(combined)   # same feature code as training
train = feat[~feat["is_future"]]
upcoming = feat[feat["is_future"]].copy()

# ------------------------------------------------------- train final models
X_train, y_train = train[FEATURES], train["dnf"].astype(int)

logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)

gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
gb.fit(X_train, y_train)

upcoming["p_logistic"] = logreg.predict_proba(upcoming[FEATURES])[:, 1]
upcoming["p_boosting"] = gb.predict_proba(upcoming[FEATURES])[:, 1]
upcoming["p_dnf"] = (upcoming["p_logistic"] + upcoming["p_boosting"]) / 2   # blend of both

# ------------------------------------------------------------------ report
cols = ["driver_code", "constructor_id", "grid", "p_dnf", "p_logistic", "p_boosting",
        "driver_dnf_last10", "team_dnf_last10", "team_dnf_season_prior"]
out = upcoming.sort_values("p_dnf", ascending=False)[cols]
pd.options.display.float_format = "{:.2f}".format
print(f"\n=== DNF risk ranking: {RACE_NAME} ({RACE_DATE}) ===")
print(out.to_string(index=False))

expected = upcoming["p_dnf"].sum()
typical = y_train.mean() * len(upcoming)
print(f"\nExpected number of DNFs this race (sum of probabilities): {expected:.1f}")
print(f"For reference, the historical average for a field this size is {typical:.1f}.")
out.to_csv("mexico_dnf_predictions.csv", index=False)
print("Saved to 'mexico_dnf_predictions.csv'.")
