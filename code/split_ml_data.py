import pandas as pd

input_file = "data/processed/katraj_airpulse_ml_final.csv"

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Remove unknown targets
df = df[df["target_24h"] != -1].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

# --------------------------------------------------
# Chronological split
# --------------------------------------------------

train = df[
    (df["timestamp"] >= "2024-01-01") &
    (df["timestamp"] < "2024-10-01")
].copy()

validation = df[
    (df["timestamp"] >= "2024-10-01") &
    (df["timestamp"] < "2025-01-01")
].copy()

test = df[
    (df["timestamp"] >= "2025-01-01") &
    (df["timestamp"] <= "2026-08-31 23:00:00")
].copy()

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("=" * 60)
print("FINAL CHRONOLOGICAL ML SPLIT")
print("=" * 60)

print("\nTraining set - Jan-Sep 2024")
print("Rows:", len(train))
print(train["target_24h"].value_counts().sort_index())

print("\nValidation set - Oct-Dec 2024")
print("Rows:", len(validation))
print(validation["target_24h"].value_counts().sort_index())

print("\nTest set - 2025 to Aug 2026")
print("Rows:", len(test))
print(test["target_24h"].value_counts().sort_index())

print("\nDate ranges:")
print("Train:", train["timestamp"].min(), "to", train["timestamp"].max())
print("Validation:", validation["timestamp"].min(), "to", validation["timestamp"].max())
print("Test:", test["timestamp"].min(), "to", test["timestamp"].max())

# --------------------------------------------------
# Save
# --------------------------------------------------

train.to_csv("data/processed/train.csv", index=False)
validation.to_csv("data/processed/validation.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("\nFiles saved:")
print("data/processed/train.csv")
print("data/processed/validation.csv")
print("data/processed/test.csv")