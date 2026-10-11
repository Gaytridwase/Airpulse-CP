import pandas as pd
import numpy as np

print("=" * 60)
print("AIRPULSE - GAP-AWARE FEATURE ENGINEERING")
print("=" * 60)


# --------------------------------------------------
# LOAD EPISODE-LABELED DATA
# --------------------------------------------------

input_file = "data/processed/katraj_airpulse_episodes_final.csv"

df = pd.read_csv(
    input_file,
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
# 2. IDENTIFY CONTINUOUS HOURLY SEGMENTS
# --------------------------------------------------

time_diff = df["timestamp"].diff()

# New segment whenever the gap is greater than 1 hour
df["_segment"] = (
    time_diff > pd.Timedelta(hours=1)
).cumsum()


print("\nTimestamp gaps > 1 hour:")

print(
    (time_diff > pd.Timedelta(hours=1)).sum()
)


# --------------------------------------------------
# 3. EXACT TIME-AWARE LAG FUNCTION
# --------------------------------------------------

def create_exact_lag(df, column, hours):

    lookup = (
        df.set_index("timestamp")[column]
    )

    required_timestamp = (
        df["timestamp"]
        - pd.Timedelta(hours=hours)
    )

    return required_timestamp.map(lookup)


# --------------------------------------------------
# 4. POLLUTION LAG FEATURES
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

    df[pollutant + "_lag_1h"] = create_exact_lag(
        df, pollutant, 1
    )

    df[pollutant + "_lag_3h"] = create_exact_lag(
        df, pollutant, 3
    )

    df[pollutant + "_lag_6h"] = create_exact_lag(
        df, pollutant, 6
    )

    df[pollutant + "_lag_24h"] = create_exact_lag(
        df, pollutant, 24
    )


# --------------------------------------------------
# 5. GAP-AWARE PM2.5 ROLLING FEATURES
# --------------------------------------------------

# Rolling features are calculated separately
# inside each continuous hourly segment.

df["PM2.5_rolling_mean_3h"] = (
    df.groupby("_segment")["PM2.5"]
    .transform(
        lambda x: x.rolling(
            window=3,
            min_periods=3
        ).mean()
    )
)

df["PM2.5_rolling_mean_6h"] = (
    df.groupby("_segment")["PM2.5"]
    .transform(
        lambda x: x.rolling(
            window=6,
            min_periods=6
        ).mean()
    )
)

df["PM2.5_rolling_mean_24h"] = (
    df.groupby("_segment")["PM2.5"]
    .transform(
        lambda x: x.rolling(
            window=24,
            min_periods=24
        ).mean()
    )
)

df["PM2.5_rolling_std_6h"] = (
    df.groupby("_segment")["PM2.5"]
    .transform(
        lambda x: x.rolling(
            window=6,
            min_periods=6
        ).std()
    )
)


# --------------------------------------------------
# 6. WEATHER LAG FEATURES
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

    df[feature + "_lag_1h"] = create_exact_lag(
        df, feature, 1
    )

    df[feature + "_lag_3h"] = create_exact_lag(
        df, feature, 3
    )

    df[feature + "_lag_6h"] = create_exact_lag(
        df, feature, 6
    )


# --------------------------------------------------
# 7. REMOVE INTERNAL SEGMENT COLUMN
# --------------------------------------------------

df = df.drop(columns=["_segment"])


# --------------------------------------------------
# 8. SAVE
# --------------------------------------------------

output_path = (
    "data/processed/katraj_airpulse_features.csv"
)

df.to_csv(
    output_path,
    index=False
)


# --------------------------------------------------
# 9. SUMMARY
# --------------------------------------------------

print("\nFEATURE ENGINEERING SUMMARY")
print("-" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nTime features:")
print(
    "hour, day_of_week, month, day_of_year, season"
)

print("\nPollution lag features:")
print(
    "1h, 3h, 6h, 24h - exact timestamp based"
)

print("\nPM2.5 rolling features:")
print(
    "3h mean, 6h mean, 24h mean, 6h standard deviation"
)

print(
    "\nRolling features reset after timestamp gaps."
)

print("\nWeather lag features:")
print(
    "1h, 3h, 6h - exact timestamp based"
)

print("\nSaved to:")
print(output_path)

print("=" * 60)
print("GAP-AWARE FEATURE ENGINEERING COMPLETE")
print("=" * 60)