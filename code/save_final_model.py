import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


print("=" * 60)
print("SAVING FINAL AIRPULSE RANDOM FOREST MODEL")
print("=" * 60)


# --------------------------------------------------
# 1. Load training data
# --------------------------------------------------

train = pd.read_csv(
    "data/processed/train.csv"
)

target = "target_24h"

drop_columns = [
    "timestamp",
    "target_24h",
    "episode_label",
    "episode_id"
]

X_train = train.drop(
    columns=drop_columns
)

y_train = train[target]


# --------------------------------------------------
# 2. Identify features
# --------------------------------------------------

categorical_columns = [
    "season"
]

numerical_columns = [
    column
    for column in X_train.columns
    if column not in categorical_columns
]


print("\nNumerical features:", len(numerical_columns))
print("Categorical features:", categorical_columns)


# --------------------------------------------------
# 3. Fit numerical imputer
# --------------------------------------------------

imputer = SimpleImputer(
    strategy="median"
)

X_train_numeric = imputer.fit_transform(
    X_train[numerical_columns]
)


# --------------------------------------------------
# 4. Fit season encoder
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

X_train_season = encoder.fit_transform(
    X_train[categorical_columns]
)


# --------------------------------------------------
# 5. Combine processed features
# --------------------------------------------------

X_train_processed = np.hstack([
    X_train_numeric,
    X_train_season
])


print("\nProcessed training shape:")
print(X_train_processed.shape)


# --------------------------------------------------
# 6. Train final Random Forest
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
# 7. Create model directory
# --------------------------------------------------

model_directory = "data/processed/models"

os.makedirs(
    model_directory,
    exist_ok=True
)


# --------------------------------------------------
# 8. Save model
# --------------------------------------------------

joblib.dump(
    model,
    f"{model_directory}/airpulse_rf_model.joblib"
)


# --------------------------------------------------
# 9. Save imputer
# --------------------------------------------------

joblib.dump(
    imputer,
    f"{model_directory}/airpulse_imputer.joblib"
)


# --------------------------------------------------
# 10. Save encoder
# --------------------------------------------------

joblib.dump(
    encoder,
    f"{model_directory}/airpulse_encoder.joblib"
)


# --------------------------------------------------
# 11. Save feature schema
# --------------------------------------------------

feature_schema = {
    "numerical_columns": numerical_columns,
    "categorical_columns": categorical_columns,
    "season_categories": [
        "Winter",
        "Summer",
        "Monsoon",
        "Post-Monsoon"
    ],
    "processed_feature_count": int(
        X_train_processed.shape[1]
    ),
    "threshold": 0.015,
    "model": "RandomForestClassifier",
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42
}

with open(
    f"{model_directory}/feature_schema.json",
    "w"
) as file:

    json.dump(
        feature_schema,
        file,
        indent=4
    )


# --------------------------------------------------
# 12. Final confirmation
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL ARTIFACTS SAVED")
print("=" * 60)

print(
    f"\nModel     : {model_directory}/airpulse_rf_model.joblib"
)

print(
    f"Imputer   : {model_directory}/airpulse_imputer.joblib"
)

print(
    f"Encoder   : {model_directory}/airpulse_encoder.joblib"
)

print(
    f"Schema    : {model_directory}/feature_schema.json"
)

print(
    f"\nFinal feature count: {X_train_processed.shape[1]}"
)

print(
    f"Prediction threshold: {feature_schema['threshold']}"
)

print("\nDone.")