import pandas as pd

print("=" * 60)
print("AIRPULSE - 24-HOUR FORECASTING TARGET")
print("=" * 60)

# --------------------------------------------------
# LOAD FINAL EPISODE-LABELLED DATASET
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/katraj_airpulse_features.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# CREATE 24-HOUR FUTURE TARGET
# --------------------------------------------------

# Create a lookup:
# timestamp -> episode status at that timestamp
episode_lookup = df.set_index("timestamp")["episode_label"]

# Find the exact timestamp 24 hours in the future
future_timestamp = df["timestamp"] + pd.Timedelta(hours=24)

# Get the episode status exactly 24 hours later
df["target_24h"] = future_timestamp.map(episode_lookup)

# --------------------------------------------------
# UNKNOWN TARGETS
# --------------------------------------------------

# If there is no observation exactly 24 hours later,
# the target is unknown.
df["target_24h"] = df["target_24h"].fillna(-1)

# If the future episode status itself is unknown,
# keep it as unknown.
df.loc[df["target_24h"] == -1, "target_24h"] = -1

# Convert target to integer
df["target_24h"] = df["target_24h"].astype(int)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nFORECASTING TARGET SUMMARY")
print("-" * 60)

print("Total rows:", len(df))

print("\nTarget counts:")
print(df["target_24h"].value_counts().sort_index())

print("\nTarget meaning:")
print("1  = Episode occurs exactly 24 hours later")
print("0  = No episode exactly 24 hours later")
print("-1 = Unknown / insufficient future data")

# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = "data/processed/katraj_airpulse_ml_final.csv"

df.to_csv(output_path, index=False)

print("\nSaved to:")
print(output_path)

print("=" * 60)
print("24-HOUR FORECASTING TARGET COMPLETE")
print("=" * 60)