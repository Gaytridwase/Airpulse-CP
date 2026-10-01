import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_cleaned.csv")

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
print("AIRPULSE - MISSING VALUE ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. Missing percentage
# --------------------------------------------------

print("\nMISSING PERCENTAGE")

for col in pollutants:
    missing = df[col].isna().sum()
    percentage = missing / len(df) * 100

    print(f"{col:8}: {missing:5} ({percentage:.2f}%)")

# --------------------------------------------------
# 2. Rows where all pollutants are missing
# --------------------------------------------------

all_missing = df[pollutants].isna().all(axis=1)

print("\nROWS WHERE ALL POLLUTANTS ARE MISSING:")
print(all_missing.sum())

# --------------------------------------------------
# 3. Rows where PM2.5 and PM10 are both missing
# --------------------------------------------------

pm_missing = df["PM2.5"].isna() & df["PM10"].isna()

print("\nROWS WHERE PM2.5 AND PM10 ARE BOTH MISSING:")
print(pm_missing.sum())

# --------------------------------------------------
# 4. Missing values by year
# --------------------------------------------------

df["year"] = df["timestamp"].dt.year

print("\nMISSING PM2.5 BY YEAR")

for year in sorted(df["year"].unique()):
    count = df.loc[
        (df["year"] == year) & df["PM2.5"].isna()
    ].shape[0]

    print(year, ":", count)

# --------------------------------------------------
# 5. Longest consecutive missing PM2.5 sequence
# --------------------------------------------------

missing = df["PM2.5"].isna()

groups = (missing != missing.shift()).cumsum()

runs = (
    df[missing]
    .groupby(groups[missing])
    .size()
)

print("\nLONGEST CONSECUTIVE PM2.5 MISSING SEQUENCE:")

if len(runs) > 0:
    print(runs.max(), "hours")
else:
    print("0 hours")

# --------------------------------------------------
# 6. Longest consecutive missing OZONE sequence
# --------------------------------------------------

missing = df["OZONE"].isna()

groups = (missing != missing.shift()).cumsum()

runs = (
    df[missing]
    .groupby(groups[missing])
    .size()
)

print("\nLONGEST CONSECUTIVE OZONE MISSING SEQUENCE:")

if len(runs) > 0:
    print(runs.max(), "hours")
else:
    print("0 hours")

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS COMPLETE")
print("=" * 60)