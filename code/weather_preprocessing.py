import pandas as pd
import numpy as np

# Load weather files
temp = pd.read_csv("data/raw/weather_temperature.csv")
pressure = pd.read_csv("data/raw/weather_pressure.csv")
radiation = pd.read_csv("data/raw/weather_radiation.csv")
wind = pd.read_csv("data/raw/weather_wind.csv")

# Convert timestamp
temp["valid_time"] = pd.to_datetime(temp["valid_time"])
pressure["valid_time"] = pd.to_datetime(pressure["valid_time"])
radiation["valid_time"] = pd.to_datetime(radiation["valid_time"])
wind["valid_time"] = pd.to_datetime(wind["valid_time"])

# Keep only required columns
temp = temp[["valid_time", "d2m", "t2m"]]
pressure = pressure[["valid_time", "sp", "tp"]]
radiation = radiation[["valid_time", "ssrd"]]
wind = wind[["valid_time", "u10", "v10"]]

# Merge all weather data
weather = temp.merge(pressure, on="valid_time")
weather = weather.merge(radiation, on="valid_time")
weather = weather.merge(wind, on="valid_time")

# Sort by time
weather = weather.sort_values("valid_time").reset_index(drop=True)

# Convert temperature from Kelvin to Celsius
weather["temperature_C"] = weather["t2m"] - 273.15
weather["dewpoint_C"] = weather["d2m"] - 273.15

# Calculate Relative Humidity
weather["RH"] = 100 * np.exp(
    (17.625 * weather["dewpoint_C"]) /
    (243.04 + weather["dewpoint_C"])
) / np.exp(
    (17.625 * weather["temperature_C"]) /
    (243.04 + weather["temperature_C"])
)

weather["RH"] = weather["RH"].clip(0, 100)

# Convert pressure from Pa to hPa
weather["pressure_hPa"] = weather["sp"] / 100

# Convert precipitation from metres to mm
weather["precipitation_mm"] = weather["tp"] * 1000
weather["precipitation_mm"] = weather["precipitation_mm"].clip(lower=0)

# Calculate wind speed
weather["wind_speed"] = np.sqrt(
    weather["u10"]**2 + weather["v10"]**2
)

# Keep required columns
weather_final = weather[[
    "valid_time",
    "temperature_C",
    "dewpoint_C",
    "RH",
    "pressure_hPa",
    "precipitation_mm",
    "wind_speed",
    "ssrd"
]].copy()

# Rename timestamp
weather_final = weather_final.rename(
    columns={"valid_time": "timestamp"}
)

# Check missing values
print("\nMissing values:")
print(weather_final.isnull().sum())

# Display final data
print("\nFinal weather dataset:")
print(weather_final.head())

print("\nDataset shape:")
print(weather_final.shape)

# Save processed dataset
weather_final.to_csv(
    "data/processed/katraj_weather_hourly.csv",
    index=False
)

print("\nWeather preprocessing completed!")
print("File saved as: data/processed/katraj_weather_hourly.csv")