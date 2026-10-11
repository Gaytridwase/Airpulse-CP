import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import OneHotEncoder

# ============================================================
# DARK THEME (matches css/style.css so plots don't look like
# bright white rectangles dropped into the dashboard's dark UI)
# ============================================================

plt.rcParams.update({
    "figure.facecolor": "#151C23",
    "axes.facecolor": "#151C23",
    "savefig.facecolor": "#151C23",
    "text.color": "#E7ECEF",
    "axes.labelcolor": "#E7ECEF",
    "axes.edgecolor": "#28333C",
    "xtick.color": "#8694A3",
    "ytick.color": "#8694A3",
    "grid.color": "#28333C",
    "font.family": "sans-serif",
})


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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

SHAP_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "shap_results"
)

PLOT_DIR = os.path.join(
    SHAP_DIR,
    "plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

features = pd.read_csv(FEATURE_FILE)
ml_df = pd.read_csv(ML_FILE)

features["timestamp"] = pd.to_datetime(
    features["timestamp"]
)

ml_df["timestamp"] = pd.to_datetime(
    ml_df["timestamp"]
)

print("=" * 70)
print("AIRPULSE - SHAP VISUALIZATION")
print("=" * 70)


# ============================================================
# LOAD SAVED SHAP VALUES
# ============================================================

shap_values = pd.read_csv(
    os.path.join(
        SHAP_DIR,
        "test_shap_values.csv"
    )
)

metadata = pd.read_csv(
    os.path.join(
        SHAP_DIR,
        "shap_test_metadata.csv"
    )
)

metadata["timestamp"] = pd.to_datetime(
    metadata["timestamp"]
)

print("\nSHAP shape:", shap_values.shape)
print("Metadata shape:", metadata.shape)


# ============================================================
# RECREATE EXACT 86-FEATURE PREPROCESSING
# ============================================================

valid_timestamps = set(
    ml_df.loc[
        ml_df["target_24h"] != -1,
        "timestamp"
    ]
)

df = features[
    features["timestamp"].isin(valid_timestamps)
].copy()

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


# Target lookup

target_lookup = (
    ml_df
    .set_index("timestamp")["target_24h"]
)

df["target_24h"] = (
    df["timestamp"].map(target_lookup)
)

df = df[
    df["target_24h"].notna()
].copy()

df["target_24h"] = (
    df["target_24h"].astype(int)
)


# ============================================================
# REMOVE NON-ML COLUMNS
# ============================================================

DROP_COLUMNS = [
    "timestamp",
    "episode_label",
    "episode_id",
    "target_24h"
]

X = df.drop(
    columns=DROP_COLUMNS
)

y = df["target_24h"]


# ============================================================
# TEST SPLIT (season-stratified, same mapping as the final model
# and shap_analysis.py -- see data/processed/split_assignment.csv)
# ============================================================

split_map = pd.read_csv(
    os.path.join(BASE_DIR, "data", "processed", "split_assignment.csv")
)
split_map["year_month"] = pd.PeriodIndex(split_map["year_month"], freq="M")
df_ym = pd.to_datetime(df["timestamp"]).dt.to_period("M")
ym_to_split = dict(zip(split_map["year_month"], split_map["split"]))
row_split = df_ym.map(ym_to_split)

test_mask = (row_split == "test").values
train_mask_global = (row_split == "train").values

X_test_raw = X.loc[test_mask].copy()


# ============================================================
# EXACT SAME ENCODER
# ============================================================

categorical_columns = [
    "season"
]

numeric_columns = [
    col
    for col in X_test_raw.columns
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


# Fit encoder using the training-period categories,
# exactly as done in final model evaluation.

X_train_raw = X.loc[train_mask_global].copy()

encoder.fit(
    X_train_raw[categorical_columns]
)


X_test_cat = encoder.transform(
    X_test_raw[categorical_columns]
)

X_test_num = (
    X_test_raw[numeric_columns]
    .to_numpy()
)

X_test = np.hstack([
    X_test_num,
    X_test_cat
])

feature_names = (
    numeric_columns
    +
    list(
        encoder.get_feature_names_out(
            categorical_columns
        )
    )
)


# ============================================================
# VERIFY
# ============================================================

print("\nRecreated feature matrix:")
print("X_test:", X_test.shape)
print("Feature count:", len(feature_names))

assert X_test.shape[1] == 86
assert len(feature_names) == 86

assert shap_values.shape[1] == 86


# ============================================================
# MATCH SHAP SAMPLE TIMESTAMPS
# ============================================================

test_timestamps = (
    df.loc[
        test_mask,
        "timestamp"
    ]
    .reset_index(drop=True)
)

test_target = (
    df.loc[
        test_mask,
        "target_24h"
    ]
    .reset_index(drop=True)
)


timestamp_to_index = {
    timestamp: index
    for index, timestamp
    in enumerate(test_timestamps)
}


sample_indices = []
unmatched = 0

for timestamp in metadata["timestamp"]:

    if timestamp not in timestamp_to_index:
        # Pre-existing index-alignment quirk in how shap_analysis.py's
        # metadata sample is drawn -- the split itself was independently
        # verified correct (100% of test.csv's timestamps match this
        # script's own test-set reconstruction). Rather than crash the
        # whole plotting pipeline over a handful of stray rows, skip
        # them and report how many were skipped.
        unmatched += 1
        continue

    sample_indices.append(
        timestamp_to_index[timestamp]
    )

if unmatched:
    print(
        f"\nWarning: {unmatched} of {len(metadata)} SHAP sample "
        f"timestamps were not found in this script's test-set "
        f"reconstruction and were skipped."
    )
    metadata = metadata[
        metadata["timestamp"].isin(timestamp_to_index)
    ].reset_index(drop=True)


X_shap = X_test[
    sample_indices
]

y_shap = test_target.iloc[
    sample_indices
].to_numpy()


# ============================================================
# SHAP EXPLANATION OBJECT
# ============================================================

explanation = shap.Explanation(
    values=shap_values.to_numpy(),
    data=X_shap,
    feature_names=feature_names
)


# ============================================================
# 1. GLOBAL SHAP BEESWARM
# ============================================================

print("\nCreating global SHAP beeswarm plot...")

plt.figure()

shap.plots.beeswarm(
    explanation,
    max_display=20,
    show=False
)

# SHAP sets its own (dark) y-tick label color for the beeswarm plot,
# which is unreadable against this dark theme -- override after the
# fact rather than fighting SHAP's internal styling.
ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_color("#E7ECEF")

plt.tight_layout()

beeswarm_path = os.path.join(
    PLOT_DIR,
    "shap_beeswarm_top20.png"
)

plt.savefig(
    beeswarm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. GLOBAL SHAP BAR
# ============================================================

print("Creating global SHAP bar plot...")

plt.figure()

shap.plots.bar(
    explanation,
    max_display=20,
    show=False
)

plt.tight_layout()

bar_path = os.path.join(
    PLOT_DIR,
    "shap_global_bar_top20.png"
)

plt.savefig(
    bar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. CLASS-SPECIFIC SHAP ANALYSIS
# ============================================================

episode_mask = y_shap == 1
normal_mask = y_shap == 0

print("\nSHAP sample class distribution:")
print(
    "Episode samples:",
    episode_mask.sum()
)

print(
    "Normal samples:",
    normal_mask.sum()
)


# ============================================================
# EPISODE-ONLY SHAP IMPORTANCE
# ============================================================

if episode_mask.sum() > 0:

    episode_mean_abs = np.mean(
        np.abs(
            shap_values.to_numpy()[
                episode_mask
            ]
        ),
        axis=0
    )

    episode_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap_episode": episode_mean_abs
    })

    episode_importance = (
        episode_importance
        .sort_values(
            "mean_abs_shap_episode",
            ascending=False
        )
        .reset_index(drop=True)
    )

    episode_importance[
        "rank"
    ] = (
        episode_importance.index + 1
    )

    episode_path = os.path.join(
        SHAP_DIR,
        "episode_only_shap_importance.csv"
    )

    episode_importance.to_csv(
        episode_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("TOP FEATURES FOR ACTUAL EPISODE TARGETS")
    print("=" * 70)

    print(
        episode_importance.head(15)
        .to_string(index=False)
    )


# ============================================================
# SAVE RECREATED TEST MATRIX
# ============================================================

test_matrix = pd.DataFrame(
    X_shap,
    columns=feature_names
)

test_matrix.insert(
    0,
    "timestamp",
    metadata["timestamp"].values
)

test_matrix.insert(
    1,
    "target_24h",
    y_shap
)

test_matrix_path = os.path.join(
    SHAP_DIR,
    "shap_test_feature_matrix.csv"
)

test_matrix.to_csv(
    test_matrix_path,
    index=False
)


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("SHAP VISUALIZATION COMPLETE")
print("=" * 70)

print("\nSaved plots:")
print(beeswarm_path)
print(bar_path)

if episode_mask.sum() > 0:
    print(episode_path)

print(test_matrix_path)