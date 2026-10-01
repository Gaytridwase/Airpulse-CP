import pandas as pd

# Load datasets
cpcb = pd.read_csv("data/processed/katraj_cpcb_hourly.csv")
weather = pd.read_csv("data/processed/katraj_weather_hourly.csv")

# Convert timestamp to datetime
cpcb["timestamp"] = pd.to_datetime(cpcb["timestamp"])
weather["timestamp"] = pd.to_datetime(weather["timestamp"])

# Merge using timestamp
merged = pd.merge(
    cpcb,
    weather,
    on="timestamp",
    how="inner"
)

# Sort by timestamp
merged = merged.sort_values("timestamp")

# Reset index
merged = merged.reset_index(drop=True)


# Convert surface solar radiation from J/m² to W/m²
merged["solar_radiation_Wm2"] = merged["ssrd"] / 3600

# Remove original ssrd column
merged = merged.drop(columns=["ssrd"])

# Save merged dataset
output_file = "data/processed/katraj_airpulse_hourly.csv"
merged.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("CPCB + WEATHER MERGE COMPLETE")
print("=" * 60)

print("Shape:", merged.shape)

print("\nColumns:")
print(merged.columns.tolist())

print("\nFirst 5 rows:")
print(merged.head())

print("\nLast 5 rows:")
print(merged.tail())

print("\nMissing values:")
print(merged.isnull().sum())

print("\nDuplicate timestamps:", merged["timestamp"].duplicated().sum())

print("\nSaved to:", output_file)