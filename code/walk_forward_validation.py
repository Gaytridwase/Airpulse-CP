"""
Walk-forward (rolling-origin) cross-validation for AirPulse.

A single train/validation/test split -- even a season-stratified one --
is still just one roll of the dice: how well the reported metrics hold
up depends on exactly which months landed in test. Walk-forward CV
checks this by training on an expanding window and validating on the
next block, repeated across several folds spanning the full dataset,
so the final reported performance is an average (with spread) across
multiple different "futures" rather than a single lucky or unlucky one.

Fold design: start with an 8-month training window, validate on the
next 4 months, then expand the training window by those 4 months and
repeat. With ~32 months of data (Jan 2024 - Aug 2026) this produces
6 folds, each validating on a different stretch of the timeline
(different seasons, different years).
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier

THRESHOLD = 0.100
INITIAL_TRAIN_MONTHS = 8
VAL_BLOCK_MONTHS = 4

df = pd.read_csv("data/processed/katraj_airpulse_ml_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df[df["target_24h"] != -1].sort_values("timestamp").reset_index(drop=True)
df["year_month"] = df["timestamp"].dt.to_period("M")

months = sorted(df["year_month"].unique())
print(f"Total months available: {len(months)} ({months[0]} to {months[-1]})")

target = "target_24h"
drop_columns = ["timestamp", "target_24h", "episode_label", "episode_id", "year_month"]


def fit_predict(train_df, val_df, model_type):
    X_train = train_df.drop(columns=drop_columns, errors="ignore")
    X_val = val_df.drop(columns=drop_columns, errors="ignore")
    y_train = train_df[target]
    y_val = val_df[target]

    cat_cols = X_train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    imputer = SimpleImputer(strategy="median")
    Xtr_num = imputer.fit_transform(X_train[num_cols])
    Xva_num = imputer.transform(X_val[num_cols])

    encoder = OneHotEncoder(
        categories=[["Winter", "Summer", "Monsoon", "Post-Monsoon"]],
        handle_unknown="ignore", sparse_output=False
    )
    Xtr_cat = encoder.fit_transform(X_train[cat_cols])
    Xva_cat = encoder.transform(X_val[cat_cols])

    Xtr = np.hstack([Xtr_num, Xtr_cat])
    Xva = np.hstack([Xva_num, Xva_cat])

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        )
    else:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=neg / max(pos, 1), random_state=42,
            eval_metric="logloss"
        )
    model.fit(Xtr, y_train)
    proba = model.predict_proba(Xva)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)

    return {
        "val_prevalence": y_val.mean(),
        "precision": precision_score(y_val, pred, zero_division=0),
        "recall": recall_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, proba) if y_val.nunique() > 1 else np.nan,
        "pr_auc": average_precision_score(y_val, proba) if y_val.nunique() > 1 else np.nan,
    }


results = []
fold = 0
train_end = INITIAL_TRAIN_MONTHS

while train_end + VAL_BLOCK_MONTHS <= len(months):
    train_months = months[:train_end]
    val_months = months[train_end: train_end + VAL_BLOCK_MONTHS]

    train_df = df[df["year_month"].isin(train_months)]
    val_df = df[df["year_month"].isin(val_months)]

    print(f"\nFold {fold}: train {train_months[0]}..{train_months[-1]} "
          f"({len(train_df)} rows) -> validate {val_months[0]}..{val_months[-1]} "
          f"({len(val_df)} rows)")

    for model_type, label in [("rf", "Random Forest"), ("xgb", "XGBoost")]:
        metrics = fit_predict(train_df, val_df, model_type)
        metrics.update({"fold": fold, "model": label,
                         "val_start": str(val_months[0]), "val_end": str(val_months[-1])})
        print(f"  {label:14s} prevalence={metrics['val_prevalence']:.3f}  "
              f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1={metrics['f1']:.3f} ROC-AUC={metrics['roc_auc']:.3f} "
              f"PR-AUC={metrics['pr_auc']:.3f}")
        results.append(metrics)

    train_end += VAL_BLOCK_MONTHS
    fold += 1

results_df = pd.DataFrame(results)
results_df.to_csv("data/processed/walk_forward_results.csv", index=False)

print(f"\n{'='*70}\nSUMMARY ACROSS {fold} FOLDS (mean +/- std)\n{'='*70}")
summary = results_df.groupby("model")[["precision", "recall", "f1", "roc_auc", "pr_auc"]].agg(["mean", "std"])
print(summary.round(3))
print("\nSaved: data/processed/walk_forward_results.csv")
