import pandas as pd

print("=" * 60)
print("AIRPULSE - 24-HOUR FORECASTING TARGET")
print("=" * 60)

# --------------------------------------------------
# Load feature-engineered dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/katraj_airpulse_features.csv",
    parse_dates=["timestamp"]
)

df = (
    df.sort_values("timestamp")
    .reset_index(drop=True)
)

# --------------------------------------------------
# CREATE EXACT 24-HOUR FUTURE TARGET
# --------------------------------------------------
# Do NOT use shift(-24).
#
# The dataset contains timestamp gaps, so 24 rows
# does not necessarily mean 24 hours.
#
# Instead, explicitly look for:
#
# current timestamp + 24 hours
# --------------------------------------------------

episode_lookup = (
    df.set_index("timestamp")["episode_label"]
)

future_timestamp = (
    df["timestamp"] +
    pd.Timedelta(hours=24)
)

df["target_24h"] = (
    future_timestamp.map(
        episode_lookup
    )
)

# --------------------------------------------------
# Unknown future targets
# --------------------------------------------------
#
# If there is no observation exactly 24 hours later,
# the target cannot be determined.
#
# Also, if the future episode label is -1,
# the target is unknown.
# --------------------------------------------------

df["target_24h"] = (
    df["target_24h"]
    .fillna(-1)
    .astype(int)
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\nFORECASTING TARGET SUMMARY")
print("-" * 60)

print(
    "Total rows:",
    len(df)
)

print("\nTarget counts:")

print(
    df["target_24h"]
    .value_counts()
    .sort_index()
)

print("\nTarget meaning:")

print(
    "1  = Episode occurs exactly 24 hours later"
)

print(
    "0  = No episode exactly 24 hours later"
)

print(
    "-1 = Unknown / missing future observation"
)

# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = (
    "data/processed/"
    "katraj_airpulse_ml.csv"
)

df.to_csv(
    output_path,
    index=False
)

print("\nSaved to:")
print(output_path)

print("=" * 60)
print("24-HOUR FORECASTING TARGET COMPLETE")
print("=" * 60)