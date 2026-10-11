"""
Weather ablation study for AirPulse.

Question: how much does meteorology (pressure, temperature, wind,
solar radiation -- current reading + 1h/3h/6h lags) actually
contribute to 24h-ahead episode prediction, versus pollutant
history alone?

Trains Random Forest and XGBoost twice each: once with the full
feature set, once with the 16 weather-derived columns removed.
Same preprocessing, same hyperparameters, same 0.10 decision
threshold, same season-stratified split as the main pipeline --
the ONLY thing that changes between the two runs is whether
weather features are present.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

THRESHOLD = 0.015

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

target = "target_24h"
drop_columns = ["timestamp", "target_24h", "episode_label", "episode_id"]

WEATHER_COLUMNS = [
    "temperature_C", "pressure_hPa", "wind_speed", "solar_radiation_Wm2",
    "temperature_C_lag_1h", "temperature_C_lag_3h", "temperature_C_lag_6h",
    "pressure_hPa_lag_1h", "pressure_hPa_lag_3h", "pressure_hPa_lag_6h",
    "wind_speed_lag_1h", "wind_speed_lag_3h", "wind_speed_lag_6h",
    "solar_radiation_Wm2_lag_1h", "solar_radiation_Wm2_lag_3h",
    "solar_radiation_Wm2_lag_6h",
]


def prep(variant):
    extra_drop = WEATHER_COLUMNS if variant == "no_weather" else []
    X_train = train.drop(columns=drop_columns + extra_drop, errors="ignore")
    X_val = validation.drop(columns=drop_columns + extra_drop, errors="ignore")
    X_test = test.drop(columns=drop_columns + extra_drop, errors="ignore")

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    imputer = SimpleImputer(strategy="median")
    Xtr_num = imputer.fit_transform(X_train[num_cols])
    Xval_num = imputer.transform(X_val[num_cols])
    Xte_num = imputer.transform(X_test[num_cols])

    encoder = OneHotEncoder(
        categories=[["Winter", "Summer", "Monsoon", "Post-Monsoon"]],
        handle_unknown="ignore", sparse_output=False
    )
    Xtr_cat = encoder.fit_transform(X_train[cat_cols])
    Xval_cat = encoder.transform(X_val[cat_cols])
    Xte_cat = encoder.transform(X_test[cat_cols])

    Xtr = np.hstack([Xtr_num, Xtr_cat])
    Xva = np.hstack([Xval_num, Xval_cat])
    Xte = np.hstack([Xte_num, Xte_cat])
    return Xtr, Xva, Xte, len(num_cols) + len(cat_cols)


def evaluate(y_true, proba, threshold=THRESHOLD):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


y_train, y_val, y_test = train[target], validation[target], test[target]
results = []

for variant in ["with_weather", "no_weather"]:
    Xtr, Xva, Xte, n_features = prep(variant)
    print(f"\n{'='*60}\n{variant.upper()}  ({n_features} features)\n{'='*60}")

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(Xtr, y_train)
    rf_test = evaluate(y_test, rf.predict_proba(Xte)[:, 1])
    rf_val = evaluate(y_val, rf.predict_proba(Xva)[:, 1])
    print("RF  val :", {k: round(v, 3) if isinstance(v, float) else v for k, v in rf_val.items()})
    print("RF  test:", {k: round(v, 3) if isinstance(v, float) else v for k, v in rf_test.items()})
    results.append({"model": "Random Forest", "variant": variant, "split": "validation", **rf_val})
    results.append({"model": "Random Forest", "variant": variant, "split": "test", **rf_test})

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        scale_pos_weight=neg / pos, random_state=42,
        eval_metric="logloss"
    )
    xgb.fit(Xtr, y_train)
    xgb_test = evaluate(y_test, xgb.predict_proba(Xte)[:, 1])
    xgb_val = evaluate(y_val, xgb.predict_proba(Xva)[:, 1])
    print("XGB val :", {k: round(v, 3) if isinstance(v, float) else v for k, v in xgb_val.items()})
    print("XGB test:", {k: round(v, 3) if isinstance(v, float) else v for k, v in xgb_test.items()})
    results.append({"model": "XGBoost", "variant": variant, "split": "validation", **xgb_val})
    results.append({"model": "XGBoost", "variant": variant, "split": "test", **xgb_test})

results_df = pd.DataFrame(results)
results_df.to_csv("data/processed/weather_ablation_results.csv", index=False)

print(f"\n{'='*60}\nSUMMARY (test split only)\n{'='*60}")
summary = results_df[results_df["split"] == "test"][
    ["model", "variant", "precision", "recall", "f1", "roc_auc", "pr_auc"]
]
print(summary.to_string(index=False))
print("\nSaved: data/processed/weather_ablation_results.csv")
