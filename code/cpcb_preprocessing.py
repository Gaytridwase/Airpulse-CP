import pandas as pd

files = [
    "data/raw/Katraj_2024.xlsx",
    "data/raw/Katraj_2025.xlsx",
    "data/raw/Katraj_2026.xlsx"
]

dataframes = []

for file in files:
    print("\nProcessing:", file)

    df = pd.read_excel(file, skiprows=10)

    # Rename columns
    df.columns = [
        "from_time",
        "to_time",
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NH3",
        "SO2",
        "CO",
        "OZONE"
    ]

    # Convert date/time columns
    df["from_time"] = pd.to_datetime(
        df["from_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    df["to_time"] = pd.to_datetime(
        df["to_time"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # Convert pollutant columns to numeric
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

    for col in pollutants:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only required columns
    df = df[
        [
            "from_time",
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NH3",
            "SO2",
            "CO",
            "OZONE"
        ]
    ]

    dataframes.append(df)

# Combine all years
cpcb = pd.concat(dataframes, ignore_index=True)

# Rename timestamp column
cpcb = cpcb.rename(columns={"from_time": "timestamp"})

# Sort by timestamp
cpcb = cpcb.sort_values("timestamp")

# Remove duplicate timestamps
cpcb = cpcb.drop_duplicates(subset="timestamp")

# Reset index
cpcb = cpcb.reset_index(drop=True)

# Save cleaned CPCB data
output_file = "data/processed/katraj_cpcb_hourly.csv"

cpcb.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("CPCB PREPROCESSING COMPLETE")
print("=" * 60)

print("Shape:", cpcb.shape)
print("\nColumns:")
print(cpcb.columns.tolist())

print("\nFirst 5 rows:")
print(cpcb.head())

print("\nLast 5 rows:")
print(cpcb.tail())

print("\nMissing values:")
print(cpcb.isnull().sum())

print("\nDuplicate timestamps:", cpcb["timestamp"].duplicated().sum())

print("\nSaved to:", output_file)