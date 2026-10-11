import pandas as pd

# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/katraj_airpulse_cleaned.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

# --------------------------------------------------
# 1. Calculate threshold using ONLY reference period
# --------------------------------------------------

reference_df = df[
    (df["timestamp"] >= "2024-01-01") &
    (df["timestamp"] < "2024-10-01")
].copy()

threshold = reference_df[
    "PM2.5"
].dropna().quantile(0.95)

print("=" * 60)
print("AIRPULSE - EPISODE LABELING")
print("=" * 60)

print(
    f"\nReference period: "
    f"2024-01-01 to 2024-09-30"
)

print(
    f"95th percentile threshold: "
    f"{threshold:.2f} µg/m³"
)

# --------------------------------------------------
# 2. Identify valid PM2.5 readings
# --------------------------------------------------

df["valid_pm25"] = df["PM2.5"].notna()

df["above_threshold"] = (
    df["valid_pm25"] &
    (df["PM2.5"] > threshold)
)

# --------------------------------------------------
# 3. Identify continuous hourly segments
# --------------------------------------------------

time_diff = (
    df["timestamp"].diff()
    .dt.total_seconds()
    / 3600
)

# New segment whenever timestamp is not
# exactly one hour after the previous timestamp.

df["time_segment"] = (
    time_diff.ne(1)
).cumsum()

# --------------------------------------------------
# 4. Identify consecutive threshold runs
# --------------------------------------------------

episode_condition = df["above_threshold"]

run_group = (
    episode_condition
    .ne(episode_condition.shift())
    .cumsum()
)

df["run_group"] = run_group

# --------------------------------------------------
# 5. Calculate duration of each above-threshold run
# --------------------------------------------------

run_info = (
    df[df["above_threshold"]]
    .groupby(
        ["time_segment", "run_group"]
    )
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        duration_hours=("timestamp", "count")
    )
)

# --------------------------------------------------
# 6. Keep only runs lasting >= 2 valid hours
# --------------------------------------------------

episode_runs = run_info[
    run_info["duration_hours"] >= 2
]

# --------------------------------------------------
# 7. Create episode labels
# --------------------------------------------------

# Start with normal
df["episode_label"] = 0

for (
    time_segment,
    run_group_id
) in episode_runs.index:

    mask = (
        (df["time_segment"] == time_segment) &
        (df["run_group"] == run_group_id)
    )

    df.loc[
        mask,
        "episode_label"
    ] = 1

# --------------------------------------------------
# 8. Missing PM2.5 = unknown
# --------------------------------------------------

df.loc[
    df["PM2.5"].isna(),
    "episode_label"
] = -1

# --------------------------------------------------
# 9. Create episode IDs
# --------------------------------------------------

df["episode_id"] = 0

episode_number = 1

for (
    time_segment,
    run_group_id
) in episode_runs.index:

    mask = (
        (df["time_segment"] == time_segment) &
        (df["run_group"] == run_group_id)
    )

    df.loc[
        mask,
        "episode_id"
    ] = episode_number

    episode_number += 1

# --------------------------------------------------
# 10. Remove temporary columns
# --------------------------------------------------

df.drop(
    columns=[
        "valid_pm25",
        "above_threshold",
        "time_segment",
        "run_group"
    ],
    inplace=True
)

# --------------------------------------------------
# 11. Summary
# --------------------------------------------------

print("\nEPISODE SUMMARY")

print(
    "Total episodes:",
    len(episode_runs)
)

print(
    "Episode hours:",
    (df["episode_label"] == 1).sum()
)

print(
    "Normal hours:",
    (df["episode_label"] == 0).sum()
)

print(
    "Unknown hours:",
    (df["episode_label"] == -1).sum()
)

print("\nEPISODE LABEL COUNTS")

print(
    df["episode_label"]
    .value_counts()
    .sort_index()
)

# --------------------------------------------------
# 12. Save
# --------------------------------------------------

output_path = (
    "data/processed/"
    "katraj_airpulse_episodes.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 60)
print("EPISODE LABELING COMPLETE")
print("=" * 60)