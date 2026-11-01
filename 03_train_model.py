"""
STEP 3: Train and honestly evaluate DNF models.

  - Split by TIME: train on older seasons, test on newer ones (like real life).
  - Compare against simple baselines so we know if the models actually help.
  - Judge with probability-quality metrics, not plain accuracy.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES

df = pd.read_csv("f1_features.csv")

# ------------------------------------------------------ time-based split
TRAIN_THROUGH = 2022            # train on 2014-2022, test on 2023 onwards
train = df[df["season"] <= TRAIN_THROUGH]
test = df[df["season"] > TRAIN_THROUGH]
X_train, y_train = train[FEATURES], train["dnf"]
X_test, y_test = test[FEATURES], test["dnf"]

print(f"Train rows: {len(train)}  |  Test rows: {len(test)}")
print(f"DNF rate  train: {y_train.mean():.3f}  test: {y_test.mean():.3f}\n")


def score(name, probs):
    probs = np.clip(probs, 0.01, 0.99)   # avoid infinite log-loss from 0% or 100%
    return {
        "model": name,
        "log_loss": log_loss(y_test, probs),
        "brier": brier_score_loss(y_test, probs),
        "PR_AUC": average_precision_score(y_test, probs),
        "ROC_AUC": roc_auc_score(y_test, probs),
    }


results = []

# Baseline A: everyone gets the same probability (the overall training DNF rate)
results.append(score("baseline: same for all", np.full(len(test), y_train.mean())))

# Baseline B: each driver's own recent DNF rate
recent = test["driver_dnf_last10"].fillna(y_train.mean()).to_numpy()
results.append(score("baseline: driver last 10", recent))

# Model 1: logistic regression (simple and explainable)
logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                       LogisticRegression(max_iter=1000))
logreg.fit(X_train, y_train)
p_logreg = logreg.predict_proba(X_test)[:, 1]
results.append(score("logistic regression", p_logreg))

# Model 2: gradient boosting (can capture non-linear patterns; handles gaps itself)
gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                    max_iter=200, random_state=0)
gb.fit(X_train, y_train)
p_gb = gb.predict_proba(X_test)[:, 1]
results.append(score("gradient boosting", p_gb))

print(pd.DataFrame(results).round(4).to_string(index=False))
print("\nHow to read this: lower log_loss / brier = better probabilities; "
      "higher PR_AUC / ROC_AUC = better at ranking who is most at risk.")

# --------------------------------------------- is the model calibrated?
print("\nCalibration (logistic regression): when it says X%, how often do DNFs happen?")
frac_pos, mean_pred = calibration_curve(y_test, p_logreg, n_bins=6, strategy="quantile")
for predicted, actual in zip(mean_pred, frac_pos):
    print(f"  predicted {predicted:5.1%}  ->  actual {actual:5.1%}")

# ----------------------------------- the actual use case: who is most at risk?
out = test[["season", "round", "race_name", "driver_code", "grid", "status", "dnf"]].copy()
out["predicted_dnf_prob"] = p_logreg
print("\nTop 15 highest-risk driver entries in the test period:")
print(out.sort_values("predicted_dnf_prob", ascending=False).head(15).round(3).to_string(index=False))
