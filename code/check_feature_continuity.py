import pandas as pd

input_file = "data/processed/katraj_airpulse_features.csv"

df = pd.read_csv(input_file)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("=" * 60)
print("FEATURE CONTINUITY CHECK")
print("=" * 60)

# --------------------------------------------------
# 1. Find timestamp gaps
# --------------------------------------------------

time_diff = df["timestamp"].diff()

gaps = df.loc[
    time_diff > pd.Timedelta(hours=1),
    ["timestamp"]
].copy()

print("\nTimestamp gaps > 1 hour:", len(gaps))

# --------------------------------------------------
# 2. Check lag features
# --------------------------------------------------

lag_columns = [
    "PM2.5_lag_1h",
    "PM2.5_lag_3h",
    "PM2.5_lag_6h",
    "PM2.5_lag_24h",
    "PM10_lag_1h",
    "PM10_lag_3h",
    "PM10_lag_6h",
    "PM10_lag_24h"
]

print("\nLag feature columns found:")

for column in lag_columns:
    if column in df.columns:
        print("✓", column)
    else:
        print("✗", column)

# --------------------------------------------------
# 3. Check rolling features
# --------------------------------------------------

rolling_columns = [
    "PM2.5_rolling_mean_3h",
    "PM2.5_rolling_mean_6h",
    "PM2.5_rolling_mean_24h"
]

print("\nRolling feature columns found:")

for column in rolling_columns:
    if column in df.columns:
        print("✓", column)
    else:
        print("✗", column)

# --------------------------------------------------
# 4. Inspect features immediately after large gaps
# --------------------------------------------------

print("\n" + "=" * 60)
print("FEATURES AFTER TIMESTAMP GAPS")
print("=" * 60)

if len(gaps) > 0:

    gap_indices = gaps.index

    for index in gap_indices[:10]:

        print("\n----------------------------------------")
        print("Timestamp:", df.loc[index, "timestamp"])
        print("Previous timestamp:", df.loc[index - 1, "timestamp"])
        print(
            "Gap:",
            df.loc[index, "timestamp"] - df.loc[index - 1, "timestamp"]
        )

        columns_to_show = [
            "timestamp",
            "PM2.5",
            "PM2.5_lag_1h",
            "PM2.5_lag_3h",
            "PM2.5_lag_6h",
            "PM2.5_lag_24h",
            "PM2.5_rolling_mean_3h",
            "PM2.5_rolling_mean_6h",
            "PM2.5_rolling_mean_24h"
        ]

        columns_to_show = [
            column
            for column in columns_to_show
            if column in df.columns
        ]

        print(
            df.loc[
                max(0, index - 2):index + 1,
                columns_to_show
            ].to_string(index=False)
        )

print("\n" + "=" * 60)
print("CHECK COMPLETED")
print("=" * 60)