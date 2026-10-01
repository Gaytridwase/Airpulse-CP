import pandas as pd

# Load merged dataset
df = pd.read_csv("data/processed/katraj_airpulse_hourly.csv")

pollutants = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "OZONE"]

print("=" * 60)
print("AIRPULSE - OUTLIER AND DATA QUALITY ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. Basic range check
# --------------------------------------------------

print("\nMINIMUM AND MAXIMUM VALUES")

for col in pollutants:
    print(f"{col:8} Min = {df[col].min():.2f}, Max = {df[col].max():.2f}")

# --------------------------------------------------
# 2. IQR outlier analysis
# --------------------------------------------------

print("\nIQR OUTLIER ANALYSIS")

for col in pollutants:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()

    print(f"\n{col}")
    print(f"Q1       : {Q1:.2f}")
    print(f"Q3       : {Q3:.2f}")
    print(f"IQR      : {IQR:.2f}")
    print(f"Lower    : {lower:.2f}")
    print(f"Upper    : {upper:.2f}")
    print(f"Outliers : {outliers}")

# --------------------------------------------------
# 3. Show extreme pollution observations
# --------------------------------------------------

print("\nTOP 10 PM2.5 VALUES")

print(
    df[["timestamp", "PM2.5", "PM10", "NO", "NO2", "OZONE"]]
    .sort_values("PM2.5", ascending=False)
    .head(10)
)

print("\nTOP 10 PM10 VALUES")

print(
    df[["timestamp", "PM2.5", "PM10", "NO", "NO2", "OZONE"]]
    .sort_values("PM10", ascending=False)
    .head(10)
)

# --------------------------------------------------
# 4. Check suspicious zero values
# --------------------------------------------------

print("\nZERO VALUE COUNTS")

for col in pollutants:
    print(f"{col:8}: {(df[col] == 0).sum()}")

# --------------------------------------------------
# 5. Missing values
# --------------------------------------------------

print("\nMISSING VALUES")

print(df[pollutants].isnull().sum())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)