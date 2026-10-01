import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_cleaned.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

pm25 = df["PM2.5"].dropna()

print("=" * 60)
print("AIRPULSE - EPISODE THRESHOLD ANALYSIS")
print("=" * 60)

print("\nPM2.5 PERCENTILES")

percentiles = [90, 95, 97.5, 99]

for p in percentiles:
    value = pm25.quantile(p / 100)

    print(f"{p:5}th percentile : {value:.2f} µg/m³")

print("\nNUMBER OF HOURS ABOVE EACH THRESHOLD")

for p in percentiles:
    threshold = pm25.quantile(p / 100)
    count = (pm25 > threshold).sum()

    print(f"{p:5}th percentile : {count} hours")

print("\nBASIC STATISTICS")

print(f"Mean    : {pm25.mean():.2f}")
print(f"Median  : {pm25.median():.2f}")
print(f"Maximum : {pm25.max():.2f}")

print("\nVALID PM2.5 VALUES:", len(pm25))

print("\n" + "=" * 60)