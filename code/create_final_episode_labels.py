import pandas as pd

input_file = "data/processed/katraj_airpulse_features.csv"
output_file = "data/processed/katraj_airpulse_episodes_final.csv"

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# 1. Calculate threshold ONLY from Jan-Sep 2024
# --------------------------------------------------

training_period = df[
    (df["timestamp"] >= "2024-01-01") &
    (df["timestamp"] < "2024-10-01")
]

threshold = training_period["PM2.5"].dropna().quantile(0.95)

print("Threshold training period: Jan-Sep 2024")
print("95th percentile PM2.5 threshold:", threshold)

# --------------------------------------------------
# 2. Initial labels
# --------------------------------------------------

df["episode_label"] = 0

# Missing PM2.5 = unknown
df.loc[df["PM2.5"].isna(), "episode_label"] = -1

# PM2.5 above threshold = episode candidate
df.loc[df["PM2.5"] > threshold, "episode_label"] = 1

# --------------------------------------------------
# 3. Require 2 consecutive hourly observations
# --------------------------------------------------

episode = df["episode_label"].eq(1)

group = (episode != episode.shift()).cumsum()
run_length = episode.groupby(group).transform("sum")

df.loc[
    (df["episode_label"] == 1) & (run_length < 2),
    "episode_label"
] = 0

# --------------------------------------------------
# 4. Save
# --------------------------------------------------

df.to_csv(output_file, index=False)

print("\nFinal episode distribution:")
print(df["episode_label"].value_counts().sort_index())

print("\nSaved to:")
print(output_file)