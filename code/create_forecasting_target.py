import pandas as pd

print("=" * 60)
print("AIRPULSE - 24-HOUR FORECASTING TARGET")
print("=" * 60)

# Load feature-engineered dataset
df = pd.read_csv(
    "data/processed/katraj_airpulse_features.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# CREATE 24-HOUR FUTURE TARGET
# --------------------------------------------------

# Episode status 24 hours in the future
df["target_24h"] = df["episode_label"].shift(-24)

# --------------------------------------------------
# UNKNOWN TARGETS
# --------------------------------------------------

# If the future episode status is unknown (-1),
# mark the forecasting target as unknown
df.loc[df["target_24h"] == -1, "target_24h"] = -1

# Last 24 rows do not have future observations
df.loc[df.index >= len(df) - 24, "target_24h"] = -1

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nFORECASTING TARGET SUMMARY")
print("-" * 60)

print("Total rows:", len(df))

print("\nTarget counts:")
print(df["target_24h"].value_counts().sort_index())

print("\nTarget meaning:")
print("1  = Episode occurs 24 hours later")
print("0  = No episode 24 hours later")
print("-1 = Unknown / insufficient future data")

# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_path = "data/processed/katraj_airpulse_ml.csv"

df.to_csv(output_path, index=False)

print("\nSaved to:")
print(output_path)

print("=" * 60)
print("24-HOUR FORECASTING TARGET COMPLETE")
print("=" * 60)