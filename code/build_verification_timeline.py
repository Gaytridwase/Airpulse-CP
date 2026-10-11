"""
Precomputes model score + prediction + actual outcome for every test-set
row, using the season-adaptive threshold, and saves it as a flat CSV the
dashboard's Historical Verification view reads directly. Precomputing
here (rather than scoring live in Streamlit on every page load) keeps
the dashboard fast and keeps this logic in one place, consistent with
how the model is actually deployed (saved pipeline + seasonal_thresholds.json).
"""

import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = "data/processed/models"
model = joblib.load(f"{MODEL_DIR}/airpulse_rf_model.joblib")
imputer = joblib.load(f"{MODEL_DIR}/airpulse_imputer.joblib")
encoder = joblib.load(f"{MODEL_DIR}/airpulse_encoder.joblib")

with open(f"{MODEL_DIR}/feature_schema.json") as f:
    schema = json.load(f)
with open(f"{MODEL_DIR}/seasonal_thresholds.json") as f:
    seasonal_thresholds = json.load(f)

num_cols = schema["numerical_columns"]
cat_cols = schema["categorical_columns"]
global_threshold = schema["threshold"]

test = pd.read_csv("data/processed/test.csv")
test["timestamp"] = pd.to_datetime(test["timestamp"])

X_num = imputer.transform(test[num_cols])
X_cat = encoder.transform(test[cat_cols])
X = np.hstack([X_num, X_cat])

test["model_score"] = model.predict_proba(X)[:, 1]
test["threshold_used"] = test["season"].map(seasonal_thresholds).fillna(global_threshold)
test["predicted"] = (test["model_score"] >= test["threshold_used"]).astype(int)
test["actual"] = test["target_24h"]

def outcome(row):
    if row["predicted"] == 1 and row["actual"] == 1:
        return "True Positive"
    elif row["predicted"] == 1 and row["actual"] == 0:
        return "False Positive"
    elif row["predicted"] == 0 and row["actual"] == 1:
        return "False Negative"
    return "True Negative"

test["outcome"] = test.apply(outcome, axis=1)

out = test[["timestamp", "season", "model_score", "threshold_used",
            "predicted", "actual", "outcome"]].sort_values("timestamp")
out.to_csv("data/processed/verification_timeline.csv", index=False)

print(f"Saved {len(out)} rows to data/processed/verification_timeline.csv")
print(out["outcome"].value_counts())
