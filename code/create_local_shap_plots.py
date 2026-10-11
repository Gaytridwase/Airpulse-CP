import os
import pandas as pd
import matplotlib.pyplot as plt

# Dark theme matching css/style.css -- see create_shap_plots.py for the
# same block; kept here too since this script runs standalone.
plt.rcParams.update({
    "figure.facecolor": "#151C23",
    "axes.facecolor": "#151C23",
    "savefig.facecolor": "#151C23",
    "text.color": "#E7ECEF",
    "axes.labelcolor": "#E7ECEF",
    "axes.edgecolor": "#28333C",
    "xtick.color": "#8694A3",
    "ytick.color": "#8694A3",
    "grid.color": "#28333C",
    "font.family": "sans-serif",
})


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/shap_results/local_cases/"
    "all_local_shap_results.csv"
)

OUTPUT_DIR = "data/processed/shap_results/local_cases/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# CREATE PLOT FOR EACH CASE
# ============================================================

cases = ["TP", "FP", "FN"]


for case in cases:

    case_df = df[df["case"] == case].copy()

    if case_df.empty:
        print(f"No data found for {case}")
        continue

    # Take 10 strongest contributors
    case_df = case_df.sort_values(
        "abs_shap",
        ascending=False
    ).head(10)

    # Reverse for horizontal bar chart
    case_df = case_df.sort_values(
        "shap_value"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        case_df["feature"],
        case_df["shap_value"]
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    if case == "TP":
        title = "Local SHAP Explanation — True Positive"

    elif case == "FP":
        title = "Local SHAP Explanation — False Positive"

    else:
        title = "Local SHAP Explanation — False Negative"

    plt.title(title)

    plt.xlabel(
        "SHAP value (contribution to episode prediction)"
    )

    plt.ylabel("Feature")

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"local_shap_{case.lower()}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("LOCAL SHAP PLOTS COMPLETE")
print("=" * 70)

print("\nSaved in:")
print(OUTPUT_DIR)