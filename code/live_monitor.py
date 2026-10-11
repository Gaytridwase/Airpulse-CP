"""
AirPulse live monitoring module.

Pulls recent + current hourly air quality and weather data for Katraj,
Pune from Open-Meteo's free, no-API-key endpoints, builds the same lag/
rolling/time features used in training, and returns a single ready-to-
score row for the saved model pipeline.

WHY OPEN-METEO, NOT CPCB DIRECTLY: CPCB doesn't expose an open real-time
API without requesting access (data.gov.in gates it), and the project's
training data (ERA5-Land) has a multi-day delay unsuitable for "right
now" monitoring. Open-Meteo's Air Quality API (CAMS-based) and Weather
API both require no key, no signup, and cover Pune, making them the
practical choice for a working live-monitoring demo. A real deployment
serving the public would likely want to pair this with an actual CPCB/
CAAQMS feed once access is arranged -- see the note in get_live_feature_row().

KNOWN LIMITATIONS (documented, not hidden):
  - NO (nitric oxide) isn't available from Open-Meteo's free tier. Left
    as NaN; the model's saved median imputer handles it the same way it
    handles any other missing reading.
  - IQR outlier-flag and pm_consistency_flag features were computed
    offline against the FULL training distribution during feature
    engineering. There isn't a meaningful live equivalent from a few
    days of data, so they default to 0 (not flagged) for live rows --
    a documented simplification, not a hidden one.
  - Open-Meteo's CAMS pollutant model is a GLOBAL atmospheric model, not
    a ground sensor reading at Katraj specifically, so live values will
    not exactly match what a CPCB monitor at that station would show.
    This module is a working demonstration of the monitoring
    architecture, not a production-grade CPCB replacement.
"""

import numpy as np
import pandas as pd
import requests

LATITUDE = 18.47    # Katraj, Pune (matches the ERA5-Land grid cell used in training)
LONGITUDE = 73.86

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AQ_VARS = "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia"
WEATHER_VARS = (
    "temperature_2m,relative_humidity_2m,dew_point_2m,"
    "surface_pressure,precipitation,wind_speed_10m,shortwave_radiation"
)

RENAME_MAP = {
    "pm2_5": "PM2.5", "pm10": "PM10", "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2",
    "ozone": "OZONE", "ammonia": "NH3",
    "temperature_2m": "temperature_C", "relative_humidity_2m": "RH",
    "dew_point_2m": "dewpoint_C", "surface_pressure": "pressure_hPa",
    "precipitation": "precipitation_mm", "wind_speed_10m": "wind_speed",
    "shortwave_radiation": "solar_radiation_Wm2",
}

POLLUTANTS_FOR_LAGS = ["PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "OZONE"]
WEATHER_FOR_LAGS = [
    "temperature_C", "RH", "pressure_hPa",
    "precipitation_mm", "wind_speed", "solar_radiation_Wm2",
]
LAG_HOURS = [1, 3, 6, 24]


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"


def fetch_live_dataframe(past_days=3, timeout=10):
    """Fetch recent + current hourly readings. Raises requests.RequestException
    (including on no network / DNS failure / timeout) -- callers must catch
    this and show the user a clear message rather than let the page crash."""

    common = {"latitude": LATITUDE, "longitude": LONGITUDE, "timezone": "auto"}

    aq_resp = requests.get(
        AQ_URL, params={**common, "hourly": AQ_VARS, "past_days": past_days},
        timeout=timeout
    )
    aq_resp.raise_for_status()
    aq = aq_resp.json()["hourly"]

    w_resp = requests.get(
        WEATHER_URL,
        params={**common, "hourly": WEATHER_VARS, "past_days": past_days},
        timeout=timeout
    )
    w_resp.raise_for_status()
    w = w_resp.json()["hourly"]

    df = pd.merge(pd.DataFrame(aq), pd.DataFrame(w), on="time", how="inner")
    df = df.rename(columns=RENAME_MAP)
    df["timestamp"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    df["NO"] = np.nan                  # not available from Open-Meteo (see module docstring)
    df["CO"] = df["CO"] / 1000.0       # ug/m3 (Open-Meteo) -> mg/m3 (CPCB convention used in training)

    # Only keep rows up to and including the current hour -- Open-Meteo's
    # air-quality endpoint returns forecast hours too, which must not be
    # scored as if they were real observations.
    now = pd.Timestamp.now(tz=df["timestamp"].dt.tz)
    df = df[df["timestamp"] <= now]

    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(df):
    """Build the lag / rolling / time / season features AirPulse's model
    expects, from a raw hourly dataframe (as returned by fetch_live_dataframe).
    Mirrors code/feature_engineering.py's logic for the columns a live feed
    can actually supply."""

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["season"] = df["month"].apply(get_season)

    for col in POLLUTANTS_FOR_LAGS + WEATHER_FOR_LAGS:
        for h in LAG_HOURS:
            if col in ("temperature_C", "RH", "pressure_hPa",
                       "precipitation_mm", "wind_speed", "solar_radiation_Wm2") and h == 24:
                continue  # weather vars only use 1h/3h/6h lags in training
            df[f"{col}_lag_{h}h"] = df[col].shift(h)

    df["PM2.5_rolling_mean_3h"] = df["PM2.5"].rolling(3).mean()
    df["PM2.5_rolling_mean_6h"] = df["PM2.5"].rolling(6).mean()
    df["PM2.5_rolling_mean_24h"] = df["PM2.5"].rolling(24).mean()
    df["PM2.5_rolling_std_6h"] = df["PM2.5"].rolling(6).std()

    # Outlier/consistency flags: not meaningfully re-derivable from a short
    # live window against the full training distribution -- default to
    # "not flagged" (see module docstring).
    for col in ["PM2_5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "OZONE"]:
        df[f"{col}_iqr_flag"] = 0
    df["pm_consistency_flag"] = 0

    return df


def get_live_feature_row(past_days=3):
    """Returns (feature_row: pd.Series, raw_df: pd.DataFrame, latest_timestamp).
    feature_row has exactly the columns the saved model pipeline expects,
    for its most recent fully-computable hour (lag_24h needs 24h of history,
    so the usable row is a few hours behind the very latest fetch).

    Raises requests.RequestException on fetch failure.
    """

    raw = fetch_live_dataframe(past_days=past_days)
    featured = build_features(raw)

    # The most recent row with a complete 24h lag is the latest one we can
    # actually score -- later rows exist but are missing lag_24h/rolling_24h.
    usable = featured.dropna(subset=["PM2.5_lag_24h", "PM2.5_rolling_mean_24h"])

    if usable.empty:
        raise ValueError(
            "Not enough historical hours returned to compute 24h lag "
            "features yet -- try again shortly, or increase past_days."
        )

    latest_row = usable.iloc[-1]
    return latest_row, raw, latest_row["timestamp"]
