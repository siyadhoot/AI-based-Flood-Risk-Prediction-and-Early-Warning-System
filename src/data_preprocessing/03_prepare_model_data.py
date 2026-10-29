
from pathlib import Path
import pandas as pd

# Get the project folder path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output file paths
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_model_data.csv"


def main():
    # Step 1: Load the combined dataset
    df = pd.read_csv(INPUT_PATH)

    print("Original dataset shape:", df.shape)

    # Step 2: Remove duplicate events, if any
    # Keep the original row index so the target stays aligned
    df = df.loc[~df["EventID"].duplicated()].copy()

    # Step 3: Save the target column separately
    target = df["Flood Type"].copy()

    # Step 4: Remove columns that should not be model features
    exclude_columns = [
        "EventID",
        "Start Date",
        "End Date",
        "Peak Flood Level (m)",
        "Peak FL Date",
        "Num Peak FL",
        "Peak Discharge Q (cumec)",
        "Peak Discharge Date",
        "Flood Volume (cumec)",
        "Event Duration (days)",
        "Time to Peak (days)",
        "Recession Time (day)",
        "Flood Type",
        "GaugeID",
        "Start_date",
        "End_date",
        "Station",
        "River Name/ Tributory/ SubTributory",
        "Warning Level",
        "Danger Level",
        "Reliability",
        "Privacy",
        "Level_Entries",
        "Streamflow_Entries",
    ]

    X = df.drop(columns=exclude_columns, errors="ignore").copy()

    # Step 5: Remove columns that are completely empty
    all_missing = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=all_missing)

    # Step 6: Remove features with more than 50% missing values
    missing_ratio = X.isna().mean()

    mostly_missing = missing_ratio[
        missing_ratio > 0.50
    ].index.tolist()

    X = X.drop(columns=mostly_missing)

    # Step 7: Add the target column back
    # Original row indices are preserved, so labels stay aligned
    model_df = X.copy()
    model_df["Flood Type"] = target

    # Step 8: Create the output folder if it doesn't exist
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 9: Save the prepared dataset
    model_df.to_csv(OUTPUT_PATH, index=False)

    # Step 10: Display the results
    print("\n--- Model Dataset Preparation Complete ---")
    print("Final dataset shape:", model_df.shape)
    print("Number of features:", len(model_df.columns) - 1)
    print("Completely empty columns removed:", len(all_missing))
    print("Columns with more than 50% missing values removed:", len(mostly_missing))
    print("Missing cells remaining:", model_df.isna().sum().sum())

    print("\nFlood Type distribution:")
    print(model_df["Flood Type"].value_counts())

    print("\nFinal dataset saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
