import pandas as pd
import numpy as np

# Load merged dataset
df = pd.read_csv("data/processed/katraj_airpulse_hourly.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

pollutants = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NH3",
    "SO2",
    "CO",
    "OZONE"
]

print("=" * 60)
print("AIRPULSE - DATA CLEANING")
print("=" * 60)

# --------------------------------------------------
# 1. Negative values
# --------------------------------------------------

print("\nNEGATIVE VALUES")

for col in pollutants:
    count = (df[col] < 0).sum()
    print(f"{col:8}: {count}")

for col in pollutants:
    df.loc[df[col] < 0, col] = np.nan

# --------------------------------------------------
# 2. PM2.5 > PM10 consistency check
# --------------------------------------------------

invalid_pm = (
    df["PM2.5"].notna() &
    df["PM10"].notna() &
    (df["PM2.5"] > df["PM10"])
)

df["pm_consistency_flag"] = 0
df.loc[invalid_pm, "pm_consistency_flag"] = 1

print("\nPM2.5 > PM10 CASES:", invalid_pm.sum())

# --------------------------------------------------
# 3. IQR flags
# --------------------------------------------------

for col in pollutants:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    upper = Q3 + 1.5 * IQR

    flag_name = col.replace(".", "_") + "_iqr_flag"

    df[flag_name] = 0
    df.loc[df[col] > upper, flag_name] = 1

# --------------------------------------------------
# 4. Sort by timestamp
# --------------------------------------------------

df = df.sort_values("timestamp")

# --------------------------------------------------
# 5. Interpolate short missing gaps
# --------------------------------------------------

print("\nMISSING VALUES BEFORE INTERPOLATION")

print(df[pollutants].isna().sum())

for col in pollutants:

    df[col] = (
        df[col]
        .interpolate(
            method="linear",
            limit=6,
            limit_direction="both",
            limit_area="inside"
        )
    )

# --------------------------------------------------
# 6. Missing values after interpolation
# --------------------------------------------------

print("\nMISSING VALUES AFTER INTERPOLATION")

print(df[pollutants].isna().sum())

# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

output_path = "data/processed/katraj_airpulse_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nFINAL SHAPE:")
print(df.shape)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 60)
print("DATA CLEANING COMPLETE")
print("=" * 60)