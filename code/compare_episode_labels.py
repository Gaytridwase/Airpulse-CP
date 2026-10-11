import pandas as pd

# --------------------------------------------------
# Load both versions
# --------------------------------------------------

new = pd.read_csv(
    "data/processed/katraj_airpulse_episodes.csv",
    parse_dates=["timestamp"]
)

final = pd.read_csv(
    "data/processed/katraj_airpulse_episodes_final.csv",
    parse_dates=["timestamp"]
)

# --------------------------------------------------
# Compare labels
# --------------------------------------------------

comparison = new[
    ["timestamp", "PM2.5", "episode_label"]
].merge(
    final[
        ["timestamp", "PM2.5", "episode_label"]
    ],
    on="timestamp",
    suffixes=("_new", "_final")
)

different = comparison[
    comparison["episode_label_new"]
    != comparison["episode_label_final"]
].copy()

print("=" * 70)
print("EPISODE LABEL COMPARISON")
print("=" * 70)

print("\nTotal rows compared:")
print(len(comparison))

print("\nDifferent rows:")
print(len(different))

print("\nDifferences:")
print(different.to_string(index=False))

print("\n" + "=" * 70)

# --------------------------------------------------
# Show surrounding rows
# --------------------------------------------------

if len(different) > 0:

    timestamps = different["timestamp"]

    print("\nSURROUNDING DATA")
    print("=" * 70)

    for ts in timestamps:

        print(
            f"\nAround {ts}"
        )

        start = ts - pd.Timedelta(hours=3)
        end = ts + pd.Timedelta(hours=3)

        rows = new[
            (new["timestamp"] >= start) &
            (new["timestamp"] <= end)
        ][
            ["timestamp", "PM2.5", "episode_label"]
        ]

        print(rows.to_string(index=False))

print("\n" + "=" * 70)