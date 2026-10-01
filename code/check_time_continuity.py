import pandas as pd

file_path = "data/processed/katraj_airpulse_ml.csv"

df = pd.read_csv(file_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

time_diff = df["timestamp"].diff()

print("=" * 60)
print("TIME CONTINUITY CHECK")
print("=" * 60)

print("Total rows:", len(df))

print("\nTime difference counts:")
print(time_diff.value_counts().head(10))

print("\nRows exactly 1 hour apart:", (time_diff == pd.Timedelta(hours=1)).sum())
print("Rows with gaps > 1 hour:", (time_diff > pd.Timedelta(hours=1)).sum())

print("\nLargest time gaps:")
print(time_diff.sort_values(ascending=False).head(10))

print("\nDate range:")
print(df["timestamp"].min())
print(df["timestamp"].max())