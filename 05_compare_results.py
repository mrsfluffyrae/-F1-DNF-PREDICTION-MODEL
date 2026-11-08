"""
STEP 5: After the race, compare the prediction with what actually happened.

Run it AFTER `python 01_fetch_data.py` has pulled the race into your data:
    python 05_compare_results.py

It produces:
    <prefix>_comparison.csv   every driver: predicted risk vs actual outcome
    <prefix>_dnf_chart.png    a chart for your README
    <prefix>_results.md       a short summary you can paste into your README
(<prefix> comes from race.json, e.g. "brazil")
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from features import add_label

# ----------------------------------------------------------- race settings
with open("race.json") as f:
    race_cfg = json.load(f)
SEASON = race_cfg["season"]
CIRCUIT_ID = race_cfg["circuit_id"]
RACE_NAME = race_cfg["race_name"]
PREFIX = race_cfg["prefix"]
PRED_FILE = f"{PREFIX}_dnf_predictions.csv"

# ------------------------------------------------------------- load things
pred = pd.read_csv(PRED_FILE)
raw = pd.read_csv("f1_races_dataset.csv")

race = raw[(raw["season"] == SEASON) & (raw["circuit_id"] == CIRCUIT_ID)]
if race.empty:
    sys.exit("The race isn't in your data yet. Run 'python 01_fetch_data.py' again "
             "(Jolpica can take a few hours to publish results), then re-run this script.")

# Same DNF rule as in training, so the comparison is fair
actual = add_label(race)[["driver_code", "position", "status", "dnf"]]

missing = sorted(set(pred["driver_code"]) - set(actual["driver_code"]))
if missing:
    print(f"NOTE: no result found for {missing} (disqualified, or not in the data). Left out.\n")

df = pred.merge(actual, on="driver_code", how="inner")
df = df.sort_values("p_dnf", ascending=False).reset_index(drop=True)
df["risk_rank"] = df.index + 1
df["outcome"] = np.where(df["dnf"] == 1, "DNF", "Finished")

# ------------------------------------------------------------- the table
show = df[["risk_rank", "driver_code", "constructor_id", "grid", "p_dnf", "outcome", "status"]].copy()
show["p_dnf"] = (show["p_dnf"] * 100).round(1).astype(str) + "%"
print(f"=== {RACE_NAME}: prediction vs reality (riskiest first) ===")
print(show.to_string(index=False))

# ------------------------------------------------------------ the summary
n = len(df)
n_dnf = int(df["dnf"].sum())
expected = df["p_dnf"].sum()
print(f"\nRetirements: {n_dnf} actual vs {expected:.1f} expected by the model")

dnf_rows = df[df["dnf"] == 1]
ranks = ", ".join(f"{r.driver_code} (rank {r.risk_rank})" for r in dnf_rows.itertuples())
print(f"Where the retirements sat in the risk ranking: {ranks}")

for k in (3, 5, 10):
    hits = int(df.head(k)["dnf"].sum())
    print(f"Top {k} riskiest: {hits} retired ({hits / k:.0%}), compared with "
          f"{n_dnf / n:.0%} of the whole field")

# Brier score for this one race. Lower is better. The "flat" line gives everyone
# the same probability (the model's average), so beating it means the RANKING helped.
flat = np.full(n, df["p_dnf"].mean())
print("\nBrier score for this race (lower is better):")
print(f"  flat (everyone gets the average) : {brier_score_loss(df['dnf'], flat):.4f}")
for name, col in [("blend", "p_dnf"), ("logistic regression", "p_logistic"),
                  ("gradient boosting", "p_boosting")]:
    print(f"  {name:<33}: {brier_score_loss(df['dnf'], df[col]):.4f}")
if 0 < n_dnf < n:
    # (one race has few retirements, so these numbers jump around a lot)
    print("\nROC-AUC for this race (0.5 = coin flip, 1.0 = perfect ranking):")
    for name, col in [("blend", "p_dnf"), ("logistic regression", "p_logistic"),
                      ("gradient boosting", "p_boosting")]:
        print(f"  {name:<33}: {roc_auc_score(df['dnf'], df[col]):.3f}")
print("\nReminder: this is ONE race with only a handful of retirements. It's a sanity "
      "check, not proof that the model works (or doesn't).")

# ------------------------------------------------------------- save files
df.to_csv(f"{PREFIX}_comparison.csv", index=False)

top5 = int(df.head(5)["dnf"].sum())
lines = [
    f"**Actual result:** {n_dnf} drivers retired (the model expected {expected:.1f}). "
    f"Of the 5 riskiest drivers in the forecast, {top5} retired.",
    "",
    "| Risk rank | Driver | Grid | Predicted DNF | What happened |",
    "|---|---|---|---|---|",
]
for r in df[(df["risk_rank"] <= 5) | (df["dnf"] == 1)].itertuples():
    what = f"DNF ({r.status})" if r.dnf == 1 else f"Finished (P{int(r.position)})"
    lines.append(f"| {r.risk_rank} | {r.driver_code} | {int(r.grid)} | {r.p_dnf:.1%} | {what} |")
with open(f"{PREFIX}_results.md", "w") as f:
    f.write("\n".join(lines) + "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    order = df.iloc[::-1]    # riskiest at the top of the chart
    colors = np.where(order["dnf"] == 1, "#d6336c", "#b8b8b8")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(order["driver_code"], order["p_dnf"] * 100, color=colors)
    ax.set_xlabel("Predicted DNF probability (%)")
    ax.set_title(f"{RACE_NAME}: predicted risk vs what happened")
    ax.legend(handles=[Patch(color="#d6336c", label="Retired"),
                       Patch(color="#b8b8b8", label="Finished")], loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PREFIX}_dnf_chart.png", dpi=150)
    print(f"\nSaved: {PREFIX}_comparison.csv, {PREFIX}_results.md, {PREFIX}_dnf_chart.png")
except ImportError:
    print(f"\nSaved: {PREFIX}_comparison.csv, {PREFIX}_results.md")
    print("For the chart, run 'pip install matplotlib' and run this script again.")
