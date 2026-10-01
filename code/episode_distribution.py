import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_cleaned.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Station-specific threshold
threshold = df["PM2.5"].quantile(0.95)

# Above threshold
df["above_threshold"] = df["PM2.5"] > threshold

# Detect continuous groups
df["time_diff"] = df["timestamp"].diff().dt.total_seconds() / 3600

new_group = (
    (df["above_threshold"] != df["above_threshold"].shift()) |
    (df["time_diff"] != 1)
)

df["group"] = new_group.cumsum()

# Create episode groups
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

# Keep episodes >= 2 hours
episodes = episodes[episodes["duration_hours"] >= 2].copy()

episodes["year"] = episodes["start"].dt.year
episodes["month"] = episodes["start"].dt.month

print("=" * 60)
print("AIRPULSE - EPISODE DISTRIBUTION")
print("=" * 60)

print("\nTOTAL EPISODES:", len(episodes))

print("\nEPISODES BY YEAR")
print(episodes["year"].value_counts().sort_index())

print("\nEPISODE HOURS BY YEAR")
print(
    episodes.groupby("year")["duration_hours"]
    .sum()
)

print("\nEPISODES BY MONTH")
print(
    episodes["month"]
    .value_counts()
    .sort_index()
)

print("\nEPISODE DURATION STATISTICS")
print(
    episodes["duration_hours"].describe()
)

print("\n" + "=" * 60)