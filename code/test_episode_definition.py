import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_cleaned.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Calculate station-specific threshold
threshold = df["PM2.5"].quantile(0.95)

print("=" * 60)
print("AIRPULSE - EPISODE DEFINITION TEST")
print("=" * 60)

print(f"\nStation-specific PM2.5 threshold:")
print(f"95th percentile = {threshold:.2f} µg/m³")

# Above threshold
df["above_threshold"] = df["PM2.5"] > threshold

# Check consecutive hours
df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600

# A new group starts when:
# 1. threshold condition changes
# 2. there is a time gap
new_group = (
    (df["above_threshold"] != df["above_threshold"].shift()) |
    (df["time_diff"] != 1)
)

df["group"] = new_group.cumsum()

# Find groups where threshold condition is true
episodes = (
    df[df["above_threshold"]]
    .groupby("group")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        duration_hours=("timestamp", "count"),
        max_pm25=("PM2.5", "max")
    )
)

# Keep episodes lasting at least 2 hours
episodes = episodes[episodes["duration_hours"] >= 2]

print("\nNUMBER OF EPISODES:", len(episodes))

print("\nTOTAL HOURS INSIDE EPISODES:",
      episodes["duration_hours"].sum())

print("\nLONGEST EPISODE:")

if len(episodes) > 0:
    longest = episodes.sort_values(
        "duration_hours",
        ascending=False
    ).iloc[0]

    print("Start:", longest["start"])
    print("End:", longest["end"])
    print("Duration:", longest["duration_hours"], "hours")
    print("Maximum PM2.5:", longest["max_pm25"])

print("\nFIRST 10 EPISODES")

print(episodes.head(10).to_string())

print("\n" + "=" * 60)