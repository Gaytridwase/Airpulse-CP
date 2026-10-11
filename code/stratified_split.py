"""
Season-stratified chronological split for AirPulse.

v1 problem (original calendar split): validation landed almost entirely
in Pune's winter pollution season (30.3% episode prevalence) while test
spanned mostly non-winter months (1.9% prevalence). A threshold tuned on
validation collapsed on test (precision 0.45 -> 0.06).

v2 problem (first attempt at a fix, bucketing months by overall episode
RATE only): balanced overall prevalence, but by coincidence left
validation with ZERO Winter and ZERO Monsoon months, and test with ZERO
Monsoon and ZERO Post-Monsoon months. Post-Monsoon is the single most
critical season (26.4% episode rate) and it was never evaluated on test
at all -- an even more specific version of the same underlying mistake.

v3 (this version): stratify by SEASON LABEL directly, not just by raw
rate. Every season is split independently so train/validation/test each
contain at least one month of every season (where the season has enough
distinct months to do so). This is what actually guarantees the dashboard
and paper's reported test metrics reflect performance across the full
annual cycle, not just whichever seasons happened to fall where.

Post-Monsoon only has 4 total months in the whole dataset (Oct/Nov 2024,
Oct/Nov 2025), and the two years are very different (2024: 23.6% / 48.4%
rate; 2025: 0.4% / 0.0% rate -- a real year-to-year difference in winter
pollution severity, not a bug). Assigning this season chronologically
("most recent months held out") would put both 2025 near-zero months in
val/test and both 2024 high-severity months in train, which would mean
validation/test never see a genuine Post-Monsoon episode. Instead, one
high-severity 2024 month is explicitly placed in validation and one in
test, so both partitions can actually be evaluated on the season that
matters most; train keeps the two 2025 months.
"""

import pandas as pd

df = pd.read_csv("data/processed/katraj_airpulse_ml_final.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df[df["target_24h"] != -1].copy()
df = df.sort_values("timestamp").reset_index(drop=True)
df["year_month"] = df["timestamp"].dt.to_period("M")

monthly = df.groupby(["year_month", "season"])["target_24h"].agg(["size", "sum"])
monthly = monthly.reset_index().set_index("year_month")
monthly["rate"] = monthly["sum"] / monthly["size"]

assignment = {}

# Post-Monsoon: too few months (4) for a round-robin pattern. Assign by
# hand so both validation and test see real episode activity.
assignment[pd.Period("2025-10", "M")] = "train"        # 0.4% rate
assignment[pd.Period("2025-11", "M")] = "train"        # 0.0% rate
assignment[pd.Period("2024-10", "M")] = "validation"   # 23.6% rate
assignment[pd.Period("2024-11", "M")] = "test"         # 48.4% rate

# Winter, Summer, Monsoon each have 8-11 months -- a period-8 round robin
# (6 train : 1 validation : 1 test) guarantees both held-out splits get
# at least one month of every one of these seasons too.
pattern = ["train", "train", "train", "validation",
           "train", "train", "train", "test"]

for season in ["Winter", "Summer", "Monsoon"]:
    months_in_season = sorted(
        monthly[monthly["season"] == season].index.unique().tolist()
    )
    for i, ym in enumerate(months_in_season):
        assignment[ym] = pattern[i % len(pattern)]

monthly["split"] = monthly.index.map(assignment)
monthly.to_csv("data/processed/split_assignment.csv")
print(monthly[["season", "size", "sum", "rate", "split"]].to_string())

df["split"] = df["year_month"].map(assignment)

train = df[df["split"] == "train"].drop(columns=["year_month", "split"])
validation = df[df["split"] == "validation"].drop(columns=["year_month", "split"])
test = df[df["split"] == "test"].drop(columns=["year_month", "split"])

print("\n" + "=" * 70)
print("SEASON-STRATIFIED SPLIT (v3) -- every season present in every split")
print("=" * 70)
for name, part in [("Train", train), ("Validation", validation), ("Test", test)]:
    rate = part["target_24h"].mean() * 100
    print(f"\n{name}: {len(part)} rows, overall episode rate {rate:.1f}%")
    print(part.groupby("season")["target_24h"].agg(["size", "mean"]).round(3))

train.to_csv("data/processed/train.csv", index=False)
validation.to_csv("data/processed/validation.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
