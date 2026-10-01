import pandas as pd

input_file = "data/processed/katraj_airpulse_episodes_final.csv"
output_file = "data/processed/katraj_airpulse_ml_final.csv"

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

# Create timestamp -> episode label lookup
label_lookup = df[["timestamp", "episode_label"]].copy()

label_lookup["target_timestamp"] = (
    label_lookup["timestamp"] + pd.Timedelta(hours=24)
)

label_lookup = label_lookup[
    ["target_timestamp", "episode_label"]
].rename(
    columns={"episode_label": "target_24h"}
)

# Match actual observation exactly 24 hours later
df = df.merge(
    label_lookup,
    left_on="timestamp",
    right_on="target_timestamp",
    how="left"
)

df.drop(columns=["target_timestamp"], inplace=True)

# No observation exactly 24 hours later
df["target_24h"] = df["target_24h"].fillna(-1)

print("=" * 60)
print("24-HOUR FORECASTING TARGET")
print("=" * 60)

print("\nTarget distribution:")
print(df["target_24h"].value_counts().sort_index())

print("\nTotal rows:", len(df))

print("\nTarget meaning:")
print("1  = episode occurs exactly 24 hours later")
print("0  = no episode exactly 24 hours later")
print("-1 = unknown / insufficient future data")

df.to_csv(output_file, index=False)

print("\nSaved to:")
print(output_file)