import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

# --------------------------------------------------
# 2. Separate target and features
# --------------------------------------------------

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

X_test = test.drop(columns=drop_columns)
y_test = test[target]

# --------------------------------------------------
# 3. Separate categorical and numeric columns
# --------------------------------------------------

categorical_column = ["season"]

numeric_columns = [
    column for column in X_train.columns
    if column not in categorical_column
]

# --------------------------------------------------
# 4. Impute numeric features
# --------------------------------------------------

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

# --------------------------------------------------
# 5. Encode season
# --------------------------------------------------

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

# --------------------------------------------------
# 6. Combine features
# --------------------------------------------------

X_train = np.hstack(
    [X_train_numeric, X_train_season]
)

X_validation = np.hstack(
    [X_validation_numeric, X_validation_season]
)

X_test = np.hstack(
    [X_test_numeric, X_test_season]
)

# --------------------------------------------------
# 7. Train Random Forest
# --------------------------------------------------

print("=" * 60)
print("RANDOM FOREST TRAINING")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel training completed.")

# --------------------------------------------------
# 8. Validation probabilities
# --------------------------------------------------

validation_probability = model.predict_proba(
    X_validation
)[:, 1]

# --------------------------------------------------
# 9. Select operational threshold
# --------------------------------------------------

threshold = 0.20

validation_pred = (
    validation_probability >= threshold
).astype(int)

print("\n" + "=" * 60)
print("VALIDATION RESULTS - THRESHOLD 0.20")
print("=" * 60)

print("\nClassification Report:")

print(
    classification_report(
        y_validation,
        validation_pred,
        target_names=["Normal", "Episode"],
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

print("Precision:", validation_precision)
print("Recall:", validation_recall)
print("F1-score:", validation_f1)
print("ROC-AUC:", validation_roc_auc)
print("PR-AUC:", validation_pr_auc)

# --------------------------------------------------
# 10. Final test evaluation
# --------------------------------------------------

test_probability = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_probability >= threshold
).astype(int)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS - 2025 TO AUG 2026")
print("=" * 60)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_pred,
        target_names=["Normal", "Episode"],
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

print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1-score:", test_f1)
print("ROC-AUC:", test_roc_auc)
print("PR-AUC:", test_pr_auc)

# --------------------------------------------------
# 11. Feature importance
# --------------------------------------------------

feature_names = numeric_columns + list(
    encoder.get_feature_names_out(categorical_column)
)

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nTop 15 Important Features:")

print(
    importance.head(15).to_string(index=False)
)