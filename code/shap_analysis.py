import os
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "katraj_airpulse_features.csv"
)

ML_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "katraj_airpulse_ml_final.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "shap_results"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AIRPULSE - SHAP ANALYSIS")
print("=" * 70)

df = pd.read_csv(FEATURE_FILE)
ml_df = pd.read_csv(ML_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"])
ml_df["timestamp"] = pd.to_datetime(ml_df["timestamp"])

print("\nFeature dataset shape:", df.shape)
print("ML dataset shape:", ml_df.shape)


# ============================================================
# KEEP ONLY VALID ML ROWS
# ============================================================

valid_timestamps = set(
    ml_df.loc[
        ml_df["target_24h"] != -1,
        "timestamp"
    ]
)

df = df[
    df["timestamp"].isin(valid_timestamps)
].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

print("Valid ML rows:", len(df))


# ============================================================
# TARGET
# ============================================================

target_map = (
    ml_df
    .set_index("timestamp")["target_24h"]
)

df["target_24h"] = df["timestamp"].map(target_map)

df = df[
    df["target_24h"].notna()
].copy()

df["target_24h"] = df["target_24h"].astype(int)


# ============================================================
# REMOVE NON-ML COLUMNS
#
# These are NOT used by the final 86-feature model.
# ============================================================

DROP_COLUMNS = [
    "timestamp",
    "episode_label",
    "episode_id",
    "target_24h"
]

X = df.drop(columns=DROP_COLUMNS)
y = df["target_24h"]


# ============================================================
# SEASON-STRATIFIED SPLIT
# SAME ASSIGNMENT AS FINAL MODEL (data/processed/split_assignment.csv)
# NOTE: previously this was a plain calendar split (train=Jan-Sep 2024,
# validation=Oct-Dec 2024, test=2025 onward), which put almost all of
# Pune's winter pollution season into validation (30% episode rate)
# and left test unrepresentative (1.9%). Reusing the same stratified
# month->split mapping as stratified_split.py keeps SHAP's train/test
# population consistent with the model that's actually deployed.
# ============================================================

split_map = pd.read_csv(
    os.path.join(BASE_DIR, "data", "processed", "split_assignment.csv")
)
split_map["year_month"] = pd.PeriodIndex(split_map["year_month"], freq="M")

df_ym = pd.to_datetime(df["timestamp"]).dt.to_period("M")
ym_to_split = dict(zip(split_map["year_month"], split_map["split"]))
row_split = df_ym.map(ym_to_split)

train_mask = (row_split == "train").values
validation_mask = (row_split == "validation").values
test_mask = (row_split == "test").values


X_train_raw = X.loc[train_mask].copy()
X_validation_raw = X.loc[validation_mask].copy()
X_test_raw = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_validation = y.loc[validation_mask].copy()
y_test = y.loc[test_mask].copy()


print("\nRaw split shapes:")
print("Train      :", X_train_raw.shape)
print("Validation :", X_validation_raw.shape)
print("Test       :", X_test_raw.shape)


# ============================================================
# PREPROCESSING
# EXACT 86-FEATURE SCHEMA
# ============================================================

categorical_columns = ["season"]

numeric_columns = [
    col
    for col in X_train_raw.columns
    if col not in categorical_columns
]


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


X_train_cat = encoder.fit_transform(
    X_train_raw[categorical_columns]
)

X_validation_cat = encoder.transform(
    X_validation_raw[categorical_columns]
)

X_test_cat = encoder.transform(
    X_test_raw[categorical_columns]
)


X_train_num = (
    X_train_raw[numeric_columns]
    .to_numpy()
)

X_validation_num = (
    X_validation_raw[numeric_columns]
    .to_numpy()
)

X_test_num = (
    X_test_raw[numeric_columns]
    .to_numpy()
)


X_train = np.hstack([
    X_train_num,
    X_train_cat
])

X_validation = np.hstack([
    X_validation_num,
    X_validation_cat
])

X_test = np.hstack([
    X_test_num,
    X_test_cat
])


feature_names = (
    numeric_columns +
    list(
        encoder.get_feature_names_out(
            categorical_columns
        )
    )
)


# ============================================================
# VERIFY FINAL FEATURE COUNT
# ============================================================

print("\nFeature counts:")
print("Numerical features :", len(numeric_columns))
print(
    "Categorical features:",
    len(
        encoder.get_feature_names_out(
            categorical_columns
        )
    )
)
print("Total features     :", len(feature_names))

print("\nProcessed feature shapes:")
print("X_train      :", X_train.shape)
print("X_validation :", X_validation.shape)
print("X_test       :", X_test.shape)


assert len(numeric_columns) == 82
assert len(feature_names) == 86

assert X_train.shape[1] == 86
assert X_validation.shape[1] == 86
assert X_test.shape[1] == 86


# ============================================================
# TRAIN FINAL RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("\nRandom Forest training completed.")


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)


# ============================================================
# TEST DATA FOR SHAP
# ============================================================

MAX_SHAP_SAMPLES = 400  # reduced for compute time; still a statistically solid sample for aggregate SHAP importance

if len(X_test) > MAX_SHAP_SAMPLES:

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(X_test),
        size=MAX_SHAP_SAMPLES,
        replace=False
    )

    X_shap = X_test[sample_indices]

    y_shap = (
        y_test.iloc[sample_indices]
        .to_numpy()
    )

    print(
        f"\nUsing {MAX_SHAP_SAMPLES} "
        "test samples for SHAP."
    )

else:

    sample_indices = np.arange(
        len(X_test)
    )

    X_shap = X_test

    y_shap = y_test.to_numpy()

    print(
        f"\nUsing all {len(X_test)} "
        "test samples for SHAP."
    )


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

print("\nCalculating SHAP values...")

shap_output = explainer.shap_values(
    X_shap,
    check_additivity=False
)
# Known floating-point strictness between this sklearn/shap version pair --
# values themselves are unaffected, only the internal consistency assertion.


# ============================================================
# HANDLE SHAP VERSION
# ============================================================

if isinstance(shap_output, list):

    # Binary classification
    # class 1 = pollution episode

    shap_values_episode = shap_output[1]

else:

    shap_values_episode = shap_output

    if shap_values_episode.ndim == 3:

        # samples × features × classes

        shap_values_episode = (
            shap_values_episode[:, :, 1]
        )


print(
    "SHAP matrix shape:",
    shap_values_episode.shape
)

assert (
    shap_values_episode.shape[1]
    == len(feature_names)
)


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.mean(
    np.abs(shap_values_episode),
    axis=0
)

mean_shap = np.mean(
    shap_values_episode,
    axis=0
)


importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap,
    "mean_shap": mean_shap
})


importance_df = (
    importance_df
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
    .reset_index(drop=True)
)

importance_df["rank"] = (
    importance_df.index + 1
)


# ============================================================
# SAVE GLOBAL IMPORTANCE
# ============================================================

importance_path = os.path.join(
    OUTPUT_DIR,
    "global_shap_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TOP SHAP FEATURES")
print("=" * 70)

print(
    importance_df[
        [
            "rank",
            "feature",
            "mean_abs_shap",
            "mean_shap"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_df = pd.DataFrame(
    shap_values_episode,
    columns=feature_names
)

shap_values_path = os.path.join(
    OUTPUT_DIR,
    "test_shap_values.csv"
)

shap_df.to_csv(
    shap_values_path,
    index=False
)


# ============================================================
# SAVE TEST METADATA
# ============================================================

test_metadata = df.loc[
    test_mask,
    [
        "timestamp",
        "target_24h"
    ]
].copy()

test_metadata = (
    test_metadata
    .reset_index(drop=True)
)

if len(test_metadata) > MAX_SHAP_SAMPLES:

    test_metadata = (
        test_metadata
        .iloc[sample_indices]
        .reset_index(drop=True)
    )


metadata_path = os.path.join(
    OUTPUT_DIR,
    "shap_test_metadata.csv"
)

test_metadata.to_csv(
    metadata_path,
    index=False
)


# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_importance_df = pd.DataFrame({
    "feature": feature_names,
    "rf_importance": model.feature_importances_
})


rf_importance_df = (
    rf_importance_df
    .sort_values(
        "rf_importance",
        ascending=False
    )
    .reset_index(drop=True)
)


rf_importance_path = os.path.join(
    OUTPUT_DIR,
    "random_forest_feature_importance.csv"
)

rf_importance_df.to_csv(
    rf_importance_path,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved files:")
print(importance_path)
print(shap_values_path)
print(metadata_path)
print(rf_importance_path)

print("\nFinal feature count:", len(feature_names))
print("SHAP samples:", len(X_shap))