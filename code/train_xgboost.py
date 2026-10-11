import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

target = "target_24h"

drop_columns = [
    "timestamp",
    "target_24h",
    "episode_label",
    "episode_id"
]

# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X_train = train.drop(columns=drop_columns)
y_train = train[target]

X_validation = validation.drop(columns=drop_columns)
y_validation = validation[target]

X_test = test.drop(columns=drop_columns)
y_test = test[target]

# ============================================================
# 3. NUMERIC + CATEGORICAL FEATURES
# ============================================================

categorical_column = ["season"]

numeric_columns = [
    column
    for column in X_train.columns
    if column not in categorical_column
]

# ============================================================
# 4. NUMERIC IMPUTATION
# ============================================================

numeric_imputer = SimpleImputer(strategy="median")

X_train_numeric = numeric_imputer.fit_transform(
    X_train[numeric_columns]
)

X_validation_numeric = numeric_imputer.transform(
    X_validation[numeric_columns]
)

X_test_numeric = numeric_imputer.transform(
    X_test[numeric_columns]
)

# ============================================================
# 5. ONE-HOT ENCODE SEASON
# ============================================================

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

X_test_season = encoder.transform(
    X_test[categorical_column]
)

# ============================================================
# 6. COMBINE FEATURES
# ============================================================

X_train = np.hstack([
    X_train_numeric,
    X_train_season
])

X_validation = np.hstack([
    X_validation_numeric,
    X_validation_season
])

X_test = np.hstack([
    X_test_numeric,
    X_test_season
])

print("=" * 60)
print("XGBOOST TRAINING")
print("=" * 60)

print("\nFeature shapes:")
print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)
print("X_test:", X_test.shape)

# ============================================================
# 7. HANDLE CLASS IMBALANCE
# ============================================================

negative_count = np.sum(y_train == 0)
positive_count = np.sum(y_train == 1)

scale_pos_weight = negative_count / positive_count

print("\nClass distribution:")
print("Normal:", negative_count)
print("Episode:", positive_count)

print(
    "\nscale_pos_weight:",
    scale_pos_weight
)

# ============================================================
# 8. TRAIN XGBOOST
# ============================================================

model = XGBClassifier(
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

model.fit(
    X_train,
    y_train
)

print("\nModel training completed.")

# ============================================================
# 9. VALIDATION PROBABILITIES
# ============================================================

validation_probability = model.predict_proba(
    X_validation
)[:, 1]

print("\n" + "=" * 60)
print("VALIDATION PROBABILITY ANALYSIS")
print("=" * 60)

print("\nValidation probability statistics:")

print("Minimum:", validation_probability.min())
print("Maximum:", validation_probability.max())
print("Mean:", validation_probability.mean())

print("\nProbability percentiles:")

print(
    np.percentile(
        validation_probability,
        [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]
    )
)

# ============================================================
# 10. THRESHOLD ANALYSIS
# ============================================================

thresholds = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

print("\n" + "=" * 60)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 60)

print(
    "\nThreshold | Precision | Recall | F1-score"
)

print("-" * 50)

threshold_results = []

for current_threshold in thresholds:

    current_prediction = (
        validation_probability >= current_threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        current_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        current_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        current_prediction,
        zero_division=0
    )

    threshold_results.append({
        "threshold": current_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(
        f"{current_threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:8.4f}"
    )

# ============================================================
# 11. SELECT BEST VALIDATION THRESHOLD
# ============================================================

threshold_results_df = pd.DataFrame(
    threshold_results
)

best_row = threshold_results_df.loc[
    threshold_results_df["f1"].idxmax()
]

threshold = float(
    best_row["threshold"]
)

print("\nCandidate operational threshold:")

print("Threshold:", threshold)

print(
    "Validation Precision:",
    best_row["precision"]
)

print(
    "Validation Recall:",
    best_row["recall"]
)

print(
    "Validation F1:",
    best_row["f1"]
)

# ============================================================
# 12. DETAILED VALIDATION RESULTS
# ============================================================

# IMPORTANT:
# Recalculate using selected threshold.

validation_pred = (
    validation_probability >= threshold
).astype(int)

print("\n" + "=" * 60)

print(
    f"VALIDATION RESULTS - THRESHOLD {threshold:.2f}"
)

print("=" * 60)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_pred,
        target_names=[
            "Normal",
            "Episode"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_validation,
        validation_pred
    )
)

validation_precision = precision_score(
    y_validation,
    validation_pred,
    zero_division=0
)

validation_recall = recall_score(
    y_validation,
    validation_pred,
    zero_division=0
)

validation_f1 = f1_score(
    y_validation,
    validation_pred,
    zero_division=0
)

validation_roc_auc = roc_auc_score(
    y_validation,
    validation_probability
)

validation_pr_auc = average_precision_score(
    y_validation,
    validation_probability
)

print("\nValidation Metrics:")

print(
    "Precision:",
    validation_precision
)

print(
    "Recall:",
    validation_recall
)

print(
    "F1-score:",
    validation_f1
)

print(
    "ROC-AUC:",
    validation_roc_auc
)

print(
    "PR-AUC:",
    validation_pr_auc
)

# ============================================================
# 13. FINAL TEST EVALUATION
# ============================================================

test_probability = model.predict_proba(
    X_test
)[:, 1]

# Use validation-selected threshold.
# NEVER select threshold using test data.

test_pred = (
    test_probability >= threshold
).astype(int)

print("\n" + "=" * 60)

print(
    f"FINAL TEST RESULTS - 2025 TO AUG 2026 "
    f"(THRESHOLD {threshold:.2f})"
)

print("=" * 60)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=[
            "Normal",
            "Episode"
        ],
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_pred
    )
)

test_precision = precision_score(
    y_test,
    test_pred,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability
)

print("\nFinal Test Metrics:")

print(
    "Precision:",
    test_precision
)

print(
    "Recall:",
    test_recall
)

print(
    "F1-score:",
    test_f1
)

print(
    "ROC-AUC:",
    test_roc_auc
)

print(
    "PR-AUC:",
    test_pr_auc
)

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

feature_names = (
    numeric_columns
    + list(
        encoder.get_feature_names_out(
            categorical_column
        )
    )
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 15 IMPORTANT FEATURES")
print("=" * 60)

print(
    importance.head(15).to_string(
        index=False
    )
)