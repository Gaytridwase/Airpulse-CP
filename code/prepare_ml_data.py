import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

# --------------------------------------------------
# 2. Separate target
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
# 3. Handle season
# --------------------------------------------------

categorical_column = ["season"]

numeric_columns = [
    column for column in X_train.columns
    if column not in categorical_column
]

# --------------------------------------------------
# 4. Numeric missing values
#    Median fitted ONLY on training data
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
    X_train[categorical_column]
)

X_validation_season = encoder.transform(
    X_validation[categorical_column]
)

X_test_season = encoder.transform(
    X_test[categorical_column]
)

# --------------------------------------------------
# 6. Combine numeric + categorical features
# --------------------------------------------------

import numpy as np

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
# 7. Display results
# --------------------------------------------------

print("=" * 60)
print("FINAL ML DATA PREPARATION")
print("=" * 60)

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation:")
print("X_validation:", X_validation.shape)
print("y_validation:", y_validation.shape)

print("\nTest:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nOriginal numeric features:", len(numeric_columns))
print("Season categories:", encoder.categories_[0].tolist())
print("Final features:", X_train.shape[1])

print("\nMissing values after preprocessing:")
print("X_train:", np.isnan(X_train).sum())
print("X_validation:", np.isnan(X_validation).sum())
print("X_test:", np.isnan(X_test).sum())

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nValidation class distribution:")
print(y_validation.value_counts().sort_index())

print("\nTest class distribution:")
print(y_test.value_counts().sort_index())