"""
STEP 2: Turn raw results into a modelling table (label + features).
The real work happens in features.py; this script runs it and shows you
what it did so you can check it makes sense.
"""
import pandas as pd

from features import add_features, add_label, is_finisher

raw = pd.read_csv("f1_races_dataset.csv")
df = add_label(raw)

# Check the label: which statuses did we count as "finished" and which as DNF?
finisher_statuses = sorted(s for s in df["status"].unique() if is_finisher(s))
print("Statuses counted as FINISHED:", finisher_statuses[:12], "...")
print("\nMost common statuses counted as DNF:")
print(df[df["dnf"] == 1]["status"].value_counts().head(12).to_string())

print("\nShare of entries that did not finish, by season (%):")
print((df.groupby("season")["dnf"].mean() * 100).round(1).to_string())

feat = add_features(df)
feat.to_csv("f1_features.csv", index=False)

print(f"\nSaved {len(feat)} rows to 'f1_features.csv'.")
print("\nMissing values per feature (first races have no history yet; that's expected):")
cols = ["driver_dnf_last10", "driver_dnf_career", "team_dnf_last10",
        "team_dnf_season_prior", "circuit_dnf_prior"]
print(feat[cols].isna().sum().to_string())
