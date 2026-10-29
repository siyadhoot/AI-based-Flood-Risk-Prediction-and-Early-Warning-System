
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_cleaned.csv"
)


def clean_text(value):
    if pd.isna(value):
        return value
    return " ".join(str(value).split())


def main():
    df = pd.read_csv(INPUT_PATH)
    print("--- FINAL DATA CLEANING ---")
    print("Original shape:", df.shape)

    # 1. Clean extra spaces in text columns.
    text_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    for column in text_columns:
        df[column] = df[column].map(clean_text)

    # 2. Standardize Basin names consistently.
    if "Basin" in df.columns:
        df["Basin"] = df["Basin"].map(
            lambda value: value.casefold()
            if isinstance(value, str) else value
        )

    # 3. Remove exact duplicate records only.
    before = len(df)
    df = df.drop_duplicates().copy()
    print("Exact duplicate rows removed:", before - len(df))

    # 4. Keep identifiers for tracking and gauge-based splitting.
    identifiers = [
        column for column in ["EventID", "GaugeID"]
        if column in df.columns
    ]

    # 5. Exclude target leakage and fields unavailable
    #    for a pre-event prediction.
    exclude = [
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
        "EventID",
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

    feature_columns = [
        column for column in df.columns
        if column not in exclude
    ]

    X = df[feature_columns].copy()

    # 6. Remove features that are completely empty
    #    or more than 50% missing.
    missing_ratio = X.isna().mean()
    remove_columns = missing_ratio[
        (missing_ratio == 1) | (missing_ratio > 0.50)
    ].index.tolist()

    X = X.drop(columns=remove_columns)

    # 7. Combine identifiers, predictors and target.
    cleaned = df[identifiers].copy()
    cleaned = pd.concat([cleaned, X], axis=1)
    cleaned["Flood Type"] = df["Flood Type"]

    # 8. Save the cleaned dataset.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)

    print("\n--- CLEANING SUMMARY ---")
    print("Final shape:", cleaned.shape)
    print("Predictor columns:", len(X.columns))
    print("Features removed for excessive missingness:",
          len(remove_columns))
    print("Missing cells remaining:",
          int(cleaned.isna().sum().sum()))
    print("Duplicate rows remaining:",
          int(cleaned.duplicated().sum()))

    print("\nTarget distribution:")
    print(cleaned["Flood Type"].value_counts())

    print("\nSaved to:", OUTPUT_PATH)
    print("\nNote: Remaining missing values must be imputed")
    print("inside the training pipeline, fitted on training data only.")


if __name__ == "__main__":
    main()
