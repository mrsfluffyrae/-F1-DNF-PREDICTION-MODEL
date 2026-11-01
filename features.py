"""
Shared helpers. You don't run this file directly; steps 2, 3 and 4 import it.
Keeping the feature code in ONE place guarantees that training and the
final race prediction build their features in exactly the same way.
"""
import pandas as pd

# Entries that are not normal retirements, so we drop them from the data
EXCLUDE = {"Disqualified", "Excluded", "Withdrew"}

# The clues the model is allowed to use
FEATURES = [
    "grid_pos",               # starting position (pit lane start counts as 20)
    "pit_lane_start",         # 1 if started from the pit lane
    "driver_dnf_last10",      # driver's DNF rate over their previous 10 races
    "driver_dnf_career",      # driver's DNF rate over all previous races
    "driver_prior_races",     # how many races the driver has started before
    "team_dnf_last10",        # team's DNF rate over its previous 10 races
    "team_dnf_season_prior",  # team's DNF rate so far THIS season (matters in new-rules years)
    "circuit_dnf_prior",      # how many entries retired at this circuit in past years
]


def is_finisher(status):
    """Finished the race: 'Finished', 'Lapped', or '+1 Lap', '+2 Laps', ..."""
    return status in ("Finished", "Lapped") or status.startswith("+")


def add_label(df):
    """Drop odd entries and add dnf = 1 (did not finish) or 0 (finished)."""
    df = df[~df["status"].isin(EXCLUDE)].copy()
    df["dnf"] = (~df["status"].apply(is_finisher)).astype(int)
    return df


def add_features(df):
    """
    Add history-based features. Every feature for a race only uses races that
    happened BEFORE it (that's what .shift(1) does: it skips the current row).
    Rows for a future race can have dnf = NaN; they get features from the past only.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "round", "driver_id"]).reset_index(drop=True)
    df["race_idx"] = df.groupby(["season", "round"], sort=False).ngroup()

    # Starting position. grid == 0 means pit lane start; treat as last.
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    df["grid_pos"] = df["grid"].replace(0, 20)

    # Driver history
    g = df.groupby("driver_id")["dnf"]
    df["driver_dnf_last10"] = g.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    df["driver_dnf_career"] = g.transform(lambda s: s.shift(1).expanding(min_periods=3).mean())
    df["driver_prior_races"] = g.cumcount()

    # Team history: average the team's two cars per race FIRST, then look back,
    # so one car's result never leaks into its teammate's features.
    team = (df.groupby(["constructor_id", "season", "race_idx"], as_index=False)["dnf"].mean()
              .sort_values(["constructor_id", "race_idx"]))
    team["team_dnf_last10"] = (team.groupby("constructor_id")["dnf"]
                               .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()))
    team["team_dnf_season_prior"] = (team.groupby(["constructor_id", "season"])["dnf"]
                                     .transform(lambda s: s.shift(1).expanding(min_periods=2).mean()))
    df = df.merge(team[["constructor_id", "race_idx", "team_dnf_last10", "team_dnf_season_prior"]],
                  on=["constructor_id", "race_idx"], how="left")

    # Circuit history
    circ = (df.groupby(["circuit_id", "race_idx"], as_index=False)["dnf"].mean()
              .sort_values(["circuit_id", "race_idx"]))
    circ["circuit_dnf_prior"] = (circ.groupby("circuit_id")["dnf"]
                                 .transform(lambda s: s.shift(1).expanding().mean()))
    df = df.merge(circ[["circuit_id", "race_idx", "circuit_dnf_prior"]],
                  on=["circuit_id", "race_idx"], how="left")

    return df.sort_values("race_idx").reset_index(drop=True)
