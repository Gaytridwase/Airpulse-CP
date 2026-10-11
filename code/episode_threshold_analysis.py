import pandas as pd

# --------------------------------------------------
# Load cleaned dataset
# --------------------------------------------------

df = pd.read_csv(
    "data/processed/katraj_airpulse_cleaned.csv"
)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# --------------------------------------------------
# Reference period
# --------------------------------------------------
# The threshold is calculated ONLY from the
# training/reference period to avoid future leakage.
# Jan-Sep 2024 is the training period.
# --------------------------------------------------

reference_df = df[
    (df["timestamp"] >= "2024-01-01") &
    (df["timestamp"] < "2024-10-01")
].copy()

pm25 = reference_df["PM2.5"].dropna()

print("=" * 60)
print("AIRPULSE - EPISODE THRESHOLD ANALYSIS")
print("=" * 60)

print("\nREFERENCE PERIOD")
print("2024-01-01 to 2024-09-30")

print("\nPM2.5 PERCENTILES")

percentiles = [90, 95, 97.5, 99]

for p in percentiles:

    value = pm25.quantile(p / 100)

    print(
        f"{p:5}th percentile : "
        f"{value:.2f} µg/m³"
    )

print("\nNUMBER OF HOURS ABOVE EACH THRESHOLD")

for p in percentiles:

    threshold = pm25.quantile(p / 100)

    count = (
        pm25 > threshold
    ).sum()

    print(
        f"{p:5}th percentile : "
        f"{count} hours"
    )

print("\nBASIC STATISTICS")

print(
    f"Mean    : {pm25.mean():.2f}"
)

print(
    f"Median  : {pm25.median():.2f}"
)

print(
    f"Maximum : {pm25.max():.2f}"
)

print(
    "\nVALID PM2.5 VALUES:",
    len(pm25)
)

print("\n" + "=" * 60)