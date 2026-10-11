import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

print("=" * 70)
print("RANDOM FOREST - VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_csv(
    "data/processed/train.csv"
)

validation = pd.read_csv(
    "data/processed/validation.csv"
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

y_train = train[target]
y_validation = validation[target]


# --------------------------------------------------
# 3. Identify columns
# --------------------------------------------------

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = [
    col
    for col in X_train.columns
    if col not in categorical_columns
]


# --------------------------------------------------
# 4. Imputation
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


# --------------------------------------------------
# 5. Season encoding
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


# --------------------------------------------------
# 6. Combine features
# --------------------------------------------------

X_train_processed = np.hstack([
    X_train_numeric,
    X_train_categorical
])

X_validation_processed = np.hstack([
    X_validation_numeric,
    X_validation_categorical
])

print("\nFeature shapes:")
print("X_train:", X_train_processed.shape)
print("X_validation:", X_validation_processed.shape)


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
# 8. Validation probabilities
# --------------------------------------------------

validation_probability = model.predict_proba(
    X_validation_processed
)[:, 1]


print("\nValidation probability distribution:")

print(
    "Minimum:",
    validation_probability.min()
)

print(
    "Maximum:",
    validation_probability.max()
)

print(
    "Mean:",
    validation_probability.mean()
)


# --------------------------------------------------
# 9. Threshold analysis
# --------------------------------------------------

thresholds = [
    0.005,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.050,
    0.060,
    0.070,
    0.080,
    0.100
]

results = []

print("\n" + "=" * 70)
print("THRESHOLD PERFORMANCE")
print("=" * 70)

print(
    f"{'Threshold':>10}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

for threshold in thresholds:

    prediction = (
        validation_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_validation,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        prediction,
        zero_division=0
    )

    print(
        f"{threshold:>10.3f}"
        f"{precision:>12.4f}"
        f"{recall:>12.4f}"
        f"{f1:>12.4f}"
    )

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# --------------------------------------------------
# 10. Select threshold
# --------------------------------------------------
# Operational objective:
#
# Maintain validation recall >= 70%
# and among those thresholds choose the
# one with highest precision.
# --------------------------------------------------

results_df = pd.DataFrame(results)

eligible = results_df[
    results_df["recall"] >= 0.70
]

print("\n" + "=" * 70)
print("THRESHOLD SELECTION")
print("=" * 70)

if len(eligible) > 0:

    selected = eligible.sort_values(
        ["precision", "f1"],
        ascending=False
    ).iloc[0]

    print(
        "\nSelected threshold:",
        f"{selected['threshold']:.3f}"
    )

    print(
        "Validation precision:",
        f"{selected['precision']:.4f}"
    )

    print(
        "Validation recall:",
        f"{selected['recall']:.4f}"
    )

    print(
        "Validation F1:",
        f"{selected['f1']:.4f}"
    )

else:

    print(
        "\nNo tested threshold achieved "
        "validation recall >= 70%."
    )


# --------------------------------------------------
# 11. Threshold-independent metrics
# --------------------------------------------------

roc_auc = roc_auc_score(
    y_validation,
    validation_probability
)

pr_auc = average_precision_score(
    y_validation,
    validation_probability
)

print("\nThreshold-independent metrics:")

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC : {pr_auc:.4f}"
)


# --------------------------------------------------
# 12. Save results
# --------------------------------------------------

results_df.to_csv(
    "data/processed/rf_validation_thresholds.csv",
    index=False
)

print(
    "\nSaved:"
    "\ndata/processed/rf_validation_thresholds.csv"
)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)