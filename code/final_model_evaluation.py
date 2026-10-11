import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("=" * 60)
print("FINAL MODEL EVALUATION - RANDOM FOREST")
print("=" * 60)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_csv(
    "data/processed/train.csv"
)

validation = pd.read_csv(
    "data/processed/validation.csv"
)

test = pd.read_csv(
    "data/processed/test.csv"
)


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

target = "target_24h"

drop_columns = [
    "timestamp",
    "target_24h",
    "episode_label",
    "episode_id"
]

X_train = train.drop(
    columns=drop_columns,
    errors="ignore"
)

X_validation = validation.drop(
    columns=drop_columns,
    errors="ignore"
)

X_test = test.drop(
    columns=drop_columns,
    errors="ignore"
)

y_train = train[target]
y_validation = validation[target]
y_test = test[target]


# --------------------------------------------------
# 3. Identify categorical and numerical columns
# --------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = [
    col
    for col in X_train.columns
    if col not in categorical_columns
]

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of numerical features:")
print(len(numerical_columns))


# --------------------------------------------------
# 4. Median imputation
# --------------------------------------------------
# IMPORTANT:
# Imputer is fitted ONLY on training data.
# --------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train_numeric = imputer.fit_transform(
    X_train[numerical_columns]
)

X_validation_numeric = imputer.transform(
    X_validation[numerical_columns]
)

X_test_numeric = imputer.transform(
    X_test[numerical_columns]
)


# --------------------------------------------------
# 5. One-hot encoding
# --------------------------------------------------
# Explicitly define all four valid seasons.
#
# This prevents Post-Monsoon from becoming an
# unknown/all-zero category in validation/test.
# --------------------------------------------------

encoder = OneHotEncoder(
    categories=[[
        "Winter",
        "Summer",
        "Monsoon",
        "Post-Monsoon"
    ]],
    handle_unknown="ignore",
    sparse_output=False
)

X_train_categorical = encoder.fit_transform(
    X_train[categorical_columns]
)

X_validation_categorical = encoder.transform(
    X_validation[categorical_columns]
)

X_test_categorical = encoder.transform(
    X_test[categorical_columns]
)

print("\nSeason categories:")
print(
    encoder.categories_[0].tolist()
)


# --------------------------------------------------
# 6. Combine numerical + categorical features
# --------------------------------------------------

X_train_processed = np.hstack([
    X_train_numeric,
    X_train_categorical
])

X_validation_processed = np.hstack([
    X_validation_numeric,
    X_validation_categorical
])

X_test_processed = np.hstack([
    X_test_numeric,
    X_test_categorical
])

print("\nProcessed feature shapes:")

print(
    "X_train:",
    X_train_processed.shape
)

print(
    "X_validation:",
    X_validation_processed.shape
)

print(
    "X_test:",
    X_test_processed.shape
)


# --------------------------------------------------
# 7. Train Random Forest
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_processed,
    y_train
)

print("\nRandom Forest training completed.")


# --------------------------------------------------
# 8. Generate prediction scores
# --------------------------------------------------

validation_probability = model.predict_proba(
    X_validation_processed
)[:, 1]

test_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# --------------------------------------------------
# 9. Operational threshold
# --------------------------------------------------

threshold = 0.015

validation_prediction = (
    validation_probability >= threshold
).astype(int)

test_prediction = (
    test_probability >= threshold
).astype(int)


# --------------------------------------------------
# 10. Evaluation function
# --------------------------------------------------

def evaluate_model(
    name,
    y_true,
    probability,
    prediction
):

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

    roc_auc = roc_auc_score(
        y_true,
        probability
    )

    pr_auc = average_precision_score(
        y_true,
        probability
    )

    cm = confusion_matrix(
        y_true,
        prediction
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            prediction,
            zero_division=0
        )
    )

    return {
        "dataset": name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1]
    }


# --------------------------------------------------
# 11. Evaluate validation and test
# --------------------------------------------------

validation_results = evaluate_model(
    "VALIDATION RESULTS",
    y_validation,
    validation_probability,
    validation_prediction
)

test_results = evaluate_model(
    "TEST RESULTS",
    y_test,
    test_probability,
    test_prediction
)


# --------------------------------------------------
# 12. Save final metrics
# --------------------------------------------------

results = pd.DataFrame([
    validation_results,
    test_results
])

results.to_csv(
    "data/processed/final_rf_metrics.csv",
    index=False
)


# --------------------------------------------------
# 13. Dataset prevalence
# --------------------------------------------------

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

for name, y in [
    ("Train", y_train),
    ("Validation", y_validation),
    ("Test", y_test)
]:

    episode_count = int(
        (y == 1).sum()
    )

    normal_count = int(
        (y == 0).sum()
    )

    prevalence = (
        episode_count / len(y)
    ) * 100

    print(
        f"\n{name}:"
        f"\n  Normal  : {normal_count}"
        f"\n  Episode : {episode_count}"
        f"\n  Episode prevalence: {prevalence:.2f}%"
    )


print("\n" + "=" * 60)
print("FINAL EVALUATION COMPLETED")
print("=" * 60)