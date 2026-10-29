
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_model_data.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Find rows duplicated across every column
    duplicate_mask = df.duplicated(keep=False)
    duplicate_rows = df.loc[duplicate_mask].copy()

    print("Total dataset rows:", len(df))
    print("Rows involved in exact duplicates:", len(duplicate_rows))
    print(
        "Extra duplicate rows if keeping one copy:",
        df.duplicated().sum()
    )

    if duplicate_rows.empty:
        print("\nNo exact duplicate rows found.")
        return

    # Check whether duplicated feature records have conflicting labels
    feature_columns = [
        column for column in df.columns
        if column != "Flood Type"
    ]

    duplicate_features = df.duplicated(
        subset=feature_columns,
        keep=False
    )

    repeated_features = df.loc[duplicate_features].copy()

    print("\n--- Repeated feature combinations ---")
    print("Rows with repeated features:", len(repeated_features))

    label_counts = (
        repeated_features
        .groupby(feature_columns, dropna=False)["Flood Type"]
        .nunique()
    )

    conflicting_groups = label_counts[label_counts > 1]

    print(
        "Feature combinations with conflicting target labels:",
        len(conflicting_groups)
    )

    print("\n--- Sample duplicate rows ---")
    print(duplicate_rows.head(10).to_string(index=False))

    print("\n--- Target labels among exact duplicate rows ---")
    print(duplicate_rows["Flood Type"].value_counts())


if __name__ == "__main__":
    main()
