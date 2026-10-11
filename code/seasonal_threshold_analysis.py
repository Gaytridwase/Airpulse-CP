"""
Season-adaptive decision threshold for AirPulse.

Walk-forward validation showed ROC-AUC is fairly stable across time
(~0.77 average) but precision/recall at a SINGLE fixed threshold
(0.10) swing wildly fold to fold -- because episode prevalence itself
is highly seasonal:

    Monsoon        0.2% of hours are episodes
    Summer         2.8%
    Winter         8.9%
    Post-Monsoon  26.4%

A threshold tuned for one regime is miscalibrated for the others. This
script fits ONE threshold per season (using the validation set only,
same as the global threshold search in threshold_analysis_rf.py) and
checks on the untouched test set whether season-adaptive thresholds
actually beat the single global threshold.
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_DIR = "data/processed/models"
model = joblib.load(f"{MODEL_DIR}/airpulse_rf_model.joblib")
imputer = joblib.load(f"{MODEL_DIR}/airpulse_imputer.joblib")
encoder = joblib.load(f"{MODEL_DIR}/airpulse_encoder.joblib")

with open(f"{MODEL_DIR}/feature_schema.json") as f:
    schema = json.load(f)

num_cols = schema["numerical_columns"]
cat_cols = schema["categorical_columns"]
target = "target_24h"
GLOBAL_THRESHOLD = schema["threshold"]

validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")


def score(df):
    X_num = imputer.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])
    X = np.hstack([X_num, X_cat])
    return model.predict_proba(X)[:, 1]


validation = validation.copy()
test = test.copy()
validation["score"] = score(validation)
test["score"] = score(test)

SEASONS = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
threshold_grid = np.arange(0.01, 0.51, 0.01)

seasonal_thresholds = {}

print("=" * 70)
print("PER-SEASON THRESHOLD SEARCH (validation set)")
print("=" * 70)

for s in SEASONS:
    sub = validation[validation["season"] == s]
    y = sub[target]

    if y.nunique() < 2:
        # no positive examples for this season in validation ->
        # keep the global threshold rather than fit on nothing
        seasonal_thresholds[s] = GLOBAL_THRESHOLD
        print(f"{s:14s} n={len(sub):5d}  no positive cases in validation "
              f"-> keeping global threshold {GLOBAL_THRESHOLD:.3f}")
        continue

    best_f1, best_t = -1, GLOBAL_THRESHOLD
    for t in threshold_grid:
        pred = (sub["score"] >= t).astype(int)
        f1 = f1_score(y, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    seasonal_thresholds[s] = round(float(best_t), 3)
    p = precision_score(y, (sub["score"] >= best_t).astype(int), zero_division=0)
    r = recall_score(y, (sub["score"] >= best_t).astype(int), zero_division=0)
    print(f"{s:14s} n={len(sub):5d}  prevalence={y.mean():.3f}  "
          f"best_threshold={best_t:.3f}  P={p:.3f} R={r:.3f} F1={best_f1:.3f}")

with open(f"{MODEL_DIR}/seasonal_thresholds.json", "w") as f:
    json.dump(seasonal_thresholds, f, indent=4)

print("\nSaved:", f"{MODEL_DIR}/seasonal_thresholds.json")
print(json.dumps(seasonal_thresholds, indent=2))


# ------------------------------------------------------------------
# Validate the idea on TEST: global threshold vs season-adaptive
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("TEST SET: GLOBAL THRESHOLD vs SEASON-ADAPTIVE THRESHOLD")
print("=" * 70)

y_test = test[target]

global_pred = (test["score"] >= GLOBAL_THRESHOLD).astype(int)
adaptive_pred = (
    test["score"] >= test["season"].map(seasonal_thresholds).fillna(GLOBAL_THRESHOLD)
).astype(int)

for label, pred in [("Global (0.015 for everyone)", global_pred),
                     ("Season-adaptive", adaptive_pred)]:
    p = precision_score(y_test, pred, zero_division=0)
    r = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    print(f"{label:30s} P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

print("\nBreakdown by season on test:")
for s in SEASONS:
    sub = test[test["season"] == s]
    if len(sub) == 0 or sub[target].nunique() < 2:
        continue
    y = sub[target]
    g_pred = (sub["score"] >= GLOBAL_THRESHOLD).astype(int)
    a_pred = (sub["score"] >= seasonal_thresholds[s]).astype(int)
    print(f"  {s:14s} n={len(sub):5d}  "
          f"global F1={f1_score(y, g_pred, zero_division=0):.3f}  "
          f"adaptive F1={f1_score(y, a_pred, zero_division=0):.3f}  "
          f"(threshold {seasonal_thresholds[s]:.3f})")
