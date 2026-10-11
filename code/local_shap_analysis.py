import os
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/processed/shap_results/local_cases"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.015
RANDOM_STATE = 42


# ============================================================
# LOAD THE EXACT FINAL TRAIN / VALIDATION / TEST FILES
# ============================================================

train = pd.read_csv(
    "data/processed/train.csv"
)

validation = pd.read_csv(
    "data/processed/validation.csv"
)

test = pd.read_csv(
    "data/processed/test.csv"
)


# ============================================================
# SEPARATE FEATURES AND TARGET
# ============================================================

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

y_train = train[target].astype(int)
y_validation = validation[target].astype(int)
y_test = test[target].astype(int)


# ============================================================
# IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_columns = [
    col
    for col in X_train.columns
    if col not in categorical_columns
]


print("=" * 70)
print("AIRPULSE - LOCAL SHAP ANALYSIS")
print("=" * 70)

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumber of numerical features:")
print(len(numerical_columns))


# ============================================================
# EXACT SAME MEDIAN IMPUTATION
# ============================================================

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


# ============================================================
# EXACT SAME SEASON ENCODING
# ============================================================

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

season_feature_names = encoder.get_feature_names_out(
    categorical_columns
)


# ============================================================
# COMBINE FEATURES
# ============================================================

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


feature_names = (
    numerical_columns +
    list(season_feature_names)
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

X_validation_processed = pd.DataFrame(
    X_validation_processed,
    columns=feature_names
)


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


# ============================================================
# TRAIN EXACT SAME RANDOM FOREST
# ============================================================

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


# ============================================================
# TEST PREDICTIONS
# ============================================================

scores = model.predict_proba(
    X_test_processed
)[:, 1]

predictions = (
    scores >= THRESHOLD
).astype(int)


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results = test[
    ["timestamp", "target_24h"]
].copy()

results["model_score"] = scores
results["prediction"] = predictions


results["case"] = "TN"

results.loc[
    (results["target_24h"] == 1) &
    (results["prediction"] == 1),
    "case"
] = "TP"

results.loc[
    (results["target_24h"] == 0) &
    (results["prediction"] == 1),
    "case"
] = "FP"

results.loc[
    (results["target_24h"] == 1) &
    (results["prediction"] == 0),
    "case"
] = "FN"


# ============================================================
# VERIFY AGAINST FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("TEST CASE COUNTS")
print("=" * 70)

print(
    results["case"].value_counts()
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# SELECT REPRESENTATIVE CASES
# ============================================================

selected_cases = {}


# Highest-score TP
tp = results[
    results["case"] == "TP"
]

if len(tp) > 0:

    selected_cases["TP"] = tp.sort_values(
        "model_score",
        ascending=False
    ).iloc[0]


# Highest-score FP
fp = results[
    results["case"] == "FP"
]

if len(fp) > 0:

    selected_cases["FP"] = fp.sort_values(
        "model_score",
        ascending=False
    ).iloc[0]


# Highest-score FN
fn = results[
    results["case"] == "FN"
]

if len(fn) > 0:

    selected_cases["FN"] = fn.sort_values(
        "model_score",
        ascending=False
    ).iloc[0]


# Highest-score TN
tn = results[
    results["case"] == "TN"
]

if len(tn) > 0:

    selected_cases["TN"] = tn.sort_values(
        "model_score",
        ascending=False
    ).iloc[0]


# ============================================================
# EXTRACT LOCAL SHAP VALUES
# ============================================================

all_local_results = []

case_result_map = {}


for case_name, row in selected_cases.items():

    index = row.name

    x_row = X_test_processed.iloc[[index]]

    shap_output = explainer.shap_values(
        x_row
    )


    # Handle SHAP output formats
    if isinstance(shap_output, list):

        shap_values = np.asarray(
            shap_output[-1]
        )[0]

    else:

        shap_array = np.asarray(
            shap_output
        )

        if shap_array.ndim == 3:

            shap_values = (
                shap_array[0, :, -1]
            )

        elif shap_array.ndim == 2:

            shap_values = shap_array[0]

        else:

            shap_values = shap_array


    local_df = pd.DataFrame({

        "feature": feature_names,

        "feature_value":
            x_row.iloc[0].values,

        "shap_value":
            shap_values
    })


    local_df["abs_shap"] = (
        local_df["shap_value"].abs()
    )


    local_df = local_df.sort_values(
        "abs_shap",
        ascending=False
    )


    local_df["case"] = case_name

    local_df["timestamp"] = row[
        "timestamp"
    ]

    local_df["actual_target"] = row[
        "target_24h"
    ]

    local_df["model_score"] = row[
        "model_score"
    ]

    local_df["prediction"] = row[
        "prediction"
    ]


    all_local_results.append(
        local_df
    )

    case_result_map[
        case_name
    ] = local_df


    filename = (
        f"{case_name}_"
f"{pd.to_datetime(row['timestamp']).strftime('%Y%m%d_%H%M')}.csv"
    )


    local_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        index=False
    )


# ============================================================
# COMBINE ALL LOCAL RESULTS
# ============================================================

if all_local_results:

    combined = pd.concat(
        all_local_results,
        ignore_index=True
    )

    combined.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_local_shap_results.csv"
        ),
        index=False
    )


# ============================================================
# PRINT REPRESENTATIVE CASES
# ============================================================

print("\n" + "=" * 70)
print("REPRESENTATIVE CASES")
print("=" * 70)


for case_name, row in selected_cases.items():

    print("\n" + "-" * 70)

    print(case_name)

    print(
        "Timestamp:",
        row["timestamp"]
    )

    print(
        "Actual target:",
        row["target_24h"]
    )

    print(
        "Model score:",
        round(
            row["model_score"],
            6
        )
    )

    print(
        "Prediction:",
        row["prediction"]
    )


    case_result = case_result_map[
        case_name
    ]


    positive = case_result[
        case_result["shap_value"] > 0
    ].sort_values(
        "shap_value",
        ascending=False
    ).head(10)


    negative = case_result[
        case_result["shap_value"] < 0
    ].sort_values(
        "shap_value",
        ascending=True
    ).head(10)


    print(
        "\nTop factors PUSHING TOWARD episode:"
    )

    print(
        positive[
            [
                "feature",
                "feature_value",
                "shap_value"
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\nTop factors PUSHING AWAY from episode:"
    )

    print(
        negative[
            [
                "feature",
                "feature_value",
                "shap_value"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("LOCAL SHAP ANALYSIS COMPLETE")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_DIR)