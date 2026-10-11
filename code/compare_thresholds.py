import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")

target = "target_24h"

drop_columns = [
    "timestamp",
    "target_24h",
    "episode_label",
    "episode_id"
]

X_train = train.drop(columns=drop_columns)
y_train = train[target]

X_validation = validation.drop(columns=drop_columns)
y_validation = validation[target]

# ============================================================
# 2. PREPROCESSING
# ============================================================

categorical_column = ["season"]

numeric_columns = [
    column
    for column in X_train.columns
    if column not in categorical_column
]

imputer = SimpleImputer(strategy="median")

X_train_numeric = imputer.fit_transform(
    X_train[numeric_columns]
)

X_validation_numeric = imputer.transform(
    X_validation[numeric_columns]
)

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

X_train_season = encoder.fit_transform(
    X_train[categorical_column]
)

X_validation_season = encoder.transform(
    X_validation[categorical_column]
)

X_train_final = np.hstack([
    X_train_numeric,
    X_train_season
])

X_validation_final = np.hstack([
    X_validation_numeric,
    X_validation_season
])

# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train_final,
    y_train
)

rf_probability = rf.predict_proba(
    X_validation_final
)[:, 1]

print("Random Forest trained.")

# ============================================================
# 4. TRAIN XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = (
    negative_count / positive_count
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

xgb.fit(
    X_train_final,
    y_train
)

xgb_probability = xgb.predict_proba(
    X_validation_final
)[:, 1]

print("XGBoost trained.")

# ============================================================
# 5. THRESHOLD ANALYSIS FUNCTION
# ============================================================

def analyze_model(
    model_name,
    probabilities,
    y_true
):

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    thresholds = np.arange(
        0.005,
        0.201,
        0.005
    )

    results = []

    for threshold in thresholds:

        prediction = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            prediction
        ).ravel()

        alerts = int(
            np.sum(prediction)
        )

        results.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positives": fp,
            "false_negatives": fn,
            "alerts": alerts
        })

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Show all thresholds
    # --------------------------------------------------------

    print(
        "\nThreshold | Precision | Recall | F1 | "
        "False Positives | False Negatives | Alerts"
    )

    print("-" * 90)

    for _, row in results_df.iterrows():

        print(
            f"{row['threshold']:9.3f} | "
            f"{row['precision']:9.3f} | "
            f"{row['recall']:6.3f} | "
            f"{row['f1']:5.3f} | "
            f"{int(row['false_positives']):15d} | "
            f"{int(row['false_negatives']):16d} | "
            f"{int(row['alerts']):6d}"
        )

    # --------------------------------------------------------
    # Recall >= 70%
    # --------------------------------------------------------

    high_recall = results_df[
        results_df["recall"] >= 0.70
    ].copy()

    print("\n" + "-" * 70)
    print("THRESHOLDS WITH RECALL >= 70%")
    print("-" * 70)

    if len(high_recall) > 0:

        print(
            high_recall.to_string(
                index=False
            )
        )

        best_high_recall = high_recall.loc[
            high_recall["precision"].idxmax()
        ]

        print(
            "\nBest precision while maintaining "
            "recall >= 70%:"
        )

        print(
            best_high_recall.to_string()
        )

    else:

        print(
            "No threshold achieved recall >= 70%."
        )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    print("\n" + "-" * 70)
    print("BEST F1 THRESHOLD")
    print("-" * 70)

    print(
        best_f1.to_string()
    )

    return results_df


# ============================================================
# 6. RUN ANALYSIS
# ============================================================

rf_results = analyze_model(
    "RANDOM FOREST",
    rf_probability,
    y_validation
)

xgb_results = analyze_model(
    "XGBOOST",
    xgb_probability,
    y_validation
)

# ============================================================
# 7. FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL OPERATIONAL COMPARISON")
print("=" * 70)

rf_high_recall = rf_results[
    rf_results["recall"] >= 0.70
]

xgb_high_recall = xgb_results[
    xgb_results["recall"] >= 0.70
]

if len(rf_high_recall) > 0:

    rf_best = rf_high_recall.loc[
        rf_high_recall["precision"].idxmax()
    ]

    print("\nRandom Forest:")
    print(
        f"Threshold = {rf_best['threshold']:.3f}"
    )
    print(
        f"Precision = {rf_best['precision']:.3f}"
    )
    print(
        f"Recall = {rf_best['recall']:.3f}"
    )
    print(
        f"F1 = {rf_best['f1']:.3f}"
    )
    print(
        f"False Positives = "
        f"{int(rf_best['false_positives'])}"
    )
    print(
        f"False Negatives = "
        f"{int(rf_best['false_negatives'])}"
    )
    print(
        f"Alerts = {int(rf_best['alerts'])}"
    )

if len(xgb_high_recall) > 0:

    xgb_best = xgb_high_recall.loc[
        xgb_high_recall["precision"].idxmax()
    ]

    print("\nXGBoost:")
    print(
        f"Threshold = {xgb_best['threshold']:.3f}"
    )
    print(
        f"Precision = {xgb_best['precision']:.3f}"
    )
    print(
        f"Recall = {xgb_best['recall']:.3f}"
    )
    print(
        f"F1 = {xgb_best['f1']:.3f}"
    )
    print(
        f"False Positives = "
        f"{int(xgb_best['false_positives'])}"
    )
    print(
        f"False Negatives = "
        f"{int(xgb_best['false_negatives'])}"
    )
    print(
        f"Alerts = {int(xgb_best['alerts'])}"
    )