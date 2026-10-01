import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/processed/katraj_airpulse_cleaned.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# --------------------------------------------------
# 1. Calculate station-specific threshold
# --------------------------------------------------

threshold = df["PM2.5"].quantile(0.95)

print("=" * 60)
print("AIRPULSE - EPISODE LABELING")
print("=" * 60)

print(f"\n95th percentile threshold: {threshold:.2f} µg/m³")

# --------------------------------------------------
# 2. Identify readings above threshold
# --------------------------------------------------

df["above_threshold"] = df["PM2.5"] > threshold

# --------------------------------------------------
# 3. Check hourly continuity
# --------------------------------------------------

df["time_diff"] = (
    df["timestamp"].diff().dt.total_seconds() / 3600
)

# A new group starts when:
# - threshold condition changes
# - timestamp is not exactly 1 hour apart

new_group = (
    (df["above_threshold"] != df["above_threshold"].shift()) |
    (df["time_diff"] != 1)
)

df["group"] = new_group.cumsum()

# --------------------------------------------------
# 4. Find groups above threshold
# --------------------------------------------------

episode_groups = (
    df[df["above_threshold"]]
    .groupby("group")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        duration_hours=("timestamp", "count")
    )
)

# Minimum duration = 2 hours
episode_groups = episode_groups[
    episode_groups["duration_hours"] >= 2
]

# --------------------------------------------------
# 5. Create episode labels
# --------------------------------------------------

df["episode_label"] = 0

for group_id in episode_groups.index:
    df.loc[df["group"] == group_id, "episode_label"] = 1

# --------------------------------------------------
# 6. Handle missing PM2.5
# --------------------------------------------------

df.loc[df["PM2.5"].isna(), "episode_label"] = -1

# -1 means episode status cannot be determined

# --------------------------------------------------
# 7. Create episode ID
# --------------------------------------------------

df["episode_id"] = 0

episode_number = 1

for group_id in episode_groups.index:

    mask = df["group"] == group_id

    df.loc[mask, "episode_id"] = episode_number

    episode_number += 1

# --------------------------------------------------
# 8. Remove temporary columns
# --------------------------------------------------

df.drop(
    columns=["above_threshold", "time_diff", "group"],
    inplace=True
)

# --------------------------------------------------
# 9. Summary
# --------------------------------------------------

print("\nEPISODE SUMMARY")

print("Total episodes:", len(episode_groups))

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

print(df["episode_label"].value_counts().sort_index())

# --------------------------------------------------
# 10. Save
# --------------------------------------------------

output_path = (
    "data/processed/katraj_airpulse_episodes.csv"
)

df.to_csv(output_path, index=False)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 60)
print("EPISODE LABELING COMPLETE")
print("=" * 60)