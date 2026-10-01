import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_hourly.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\n" + "=" * 60)
print("AIRPULSE EDA - BASIC DATA ANALYSIS")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDate range:")
print("Start:", df["timestamp"].min())
print("End:", df["timestamp"].max())

print("\nMissing values:")
print(df.isnull().sum())

print("\nMissing percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))

print("\nDuplicate timestamps:")
print(df["timestamp"].duplicated().sum())

print("\nStatistical summary:")
print(df.describe())