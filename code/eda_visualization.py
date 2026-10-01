import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/katraj_airpulse_hourly.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

# 1. PM2.5 over time
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["PM2.5"])
plt.xlabel("Time")
plt.ylabel("PM2.5 (µg/m³)")
plt.title("PM2.5 Concentration Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. PM10 over time
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["PM10"])
plt.xlabel("Time")
plt.ylabel("PM10 (µg/m³)")
plt.title("PM10 Concentration Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Pollutant distributions
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "OZONE"]

for pollutant in pollutants:
    plt.figure(figsize=(8, 5))
    plt.hist(df[pollutant].dropna(), bins=50)
    plt.xlabel(pollutant)
    plt.ylabel("Frequency")
    plt.title(pollutant + " Distribution")
    plt.tight_layout()
    plt.show()

# 4. Average pollution by hour
df["hour"] = df["timestamp"].dt.hour

hourly = df.groupby("hour")[["PM2.5", "PM10"]].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly.index, hourly["PM2.5"], label="PM2.5")
plt.plot(hourly.index, hourly["PM10"], label="PM10")
plt.xlabel("Hour of Day")
plt.ylabel("Average Concentration")
plt.title("Average PM2.5 and PM10 by Hour")
plt.legend()
plt.tight_layout()
plt.show()

# 5. Average pollution by month
df["month"] = df["timestamp"].dt.month

monthly = df.groupby("month")[["PM2.5", "PM10"]].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly.index, monthly["PM2.5"], label="PM2.5")
plt.plot(monthly.index, monthly["PM10"], label="PM10")
plt.xlabel("Month")
plt.ylabel("Average Concentration")
plt.title("Average PM2.5 and PM10 by Month")
plt.legend()
plt.tight_layout()
plt.show()