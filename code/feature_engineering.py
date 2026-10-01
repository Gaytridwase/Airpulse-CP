import pandas as pd
import numpy as np

print("=" * 60)
print("AIRPULSE - FEATURE ENGINEERING")
print("=" * 60)

# Load episode-labeled data
df = pd.read_csv(
    "data/processed/katraj_airpulse_episodes.csv",
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# 1. TIME FEATURES
# --------------------------------------------------

df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month
df["day_of_year"] = df["timestamp"].dt.dayofyear

# Season
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df["season"] = df["month"].apply(get_season)

# --------------------------------------------------
# 2. POLLUTION LAG FEATURES
# --------------------------------------------------

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

for pollutant in pollutants:
    df[pollutant + "_lag_1h"] = df[pollutant].shift(1)
    df[pollutant + "_lag_3h"] = df[pollutant].shift(3)
    df[pollutant + "_lag_6h"] = df[pollutant].shift(6)
    df[pollutant + "_lag_24h"] = df[pollutant].shift(24)

# --------------------------------------------------
# 3. PM2.5 ROLLING FEATURES
# --------------------------------------------------

df["PM2.5_rolling_mean_3h"] = (
    df["PM2.5"].rolling(3).mean()
)

df["PM2.5_rolling_mean_6h"] = (
    df["PM2.5"].rolling(6).mean()
)

df["PM2.5_rolling_mean_24h"] = (
    df["PM2.5"].rolling(24).mean()
)

df["PM2.5_rolling_std_6h"] = (
    df["PM2.5"].rolling(6).std()
)

# --------------------------------------------------
# 4. WEATHER LAG FEATURES
# --------------------------------------------------

weather_features = [
    "temperature_C",
    "RH",
    "pressure_hPa",
    "precipitation_mm",
    "wind_speed",
    "solar_radiation_Wm2"
]

for feature in weather_features:
    df[feature + "_lag_1h"] = df[feature].shift(1)
    df[feature + "_lag_3h"] = df[feature].shift(3)
    df[feature + "_lag_6h"] = df[feature].shift(6)

# --------------------------------------------------
# 5. SAVE
# --------------------------------------------------

output_path = "data/processed/katraj_airpulse_features.csv"

df.to_csv(output_path, index=False)

print("\nFEATURE ENGINEERING SUMMARY")
print("-" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTime features:")
print("hour, day_of_week, month, day_of_year, season")

print("\nPollution lag features:")
print("1h, 3h, 6h, 24h")

print("\nPM2.5 rolling features:")
print("3h mean, 6h mean, 24h mean, 6h standard deviation")

print("\nWeather lag features:")
print("1h, 3h, 6h")

print("\nSaved to:")
print(output_path)

print("=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)