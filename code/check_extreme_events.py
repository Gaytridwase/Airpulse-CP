import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_hourly.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])

events = [
    "2024-07-14 10:00",
    "2024-11-08 02:00",
    "2026-07-16 13:00",
    "2026-07-16 14:00",
    "2025-08-10 08:00"
]

for event in events:

    event_time = pd.Timestamp(event)

    print("\n" + "=" * 70)
    print("EVENT:", event)
    print("=" * 70)

    start = event_time - pd.Timedelta(hours=3)
    end = event_time + pd.Timedelta(hours=3)

    nearby = df[
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end)
    ]

    print(
        nearby[
            [
                "timestamp",
                "PM2.5",
                "PM10",
                "NO",
                "NO2",
                "NH3",
                "SO2",
                "CO",
                "OZONE",
                "temperature_C",
                "RH",
                "wind_speed"
            ]
        ].to_string(index=False)
    )