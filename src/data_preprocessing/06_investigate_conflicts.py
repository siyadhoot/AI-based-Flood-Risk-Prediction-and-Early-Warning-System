
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "flood_model_data.csv"
COMBINED_PATH = PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"


def main():
    model_df = pd.read_csv(MODEL_PATH)
    combined_df = pd.read_csv(COMBINED_PATH)

    # Input features only; exclude the target label
    feature_columns = [
        column for column in model_df.columns
        if column != "Flood Type"
    ]

    # Find feature combinations associated with multiple labels
    label_counts = (
        model_df.groupby(feature_columns, dropna=False)["Flood Type"]
        .nunique()
    )

    conflicting_features = label_counts[label_counts > 1]

    print("Conflicting feature combinations:", len(conflicting_features))

    if conflicting_features.empty:
        print("No conflicting feature combinations found.")
        return

    # Merge the conflicting feature combinations with their labels
    conflicting_df = (
        model_df.merge(
            conflicting_features.rename("label_count").reset_index()[
                feature_columns
            ],
            on=feature_columns,
            how="inner"
        )
    )

    # Recover event identifiers and dates from the original combined data
    event_details = combined_df[
        ["EventID", "Start Date", "End Date", "Flood Type"]
    ].copy()

    # Match using the complete feature columns, not the target label
    conflict_details = conflicting_df.merge(
        combined_df,
        on=["Flood Type"] + feature_columns,
        how="left"
    )

    # Print a compact view of the conflicting records
    print("\n--- Conflicting records ---")

    columns_to_show = [
        column for column in
        ["EventID", "Start Date", "End Date", "Flood Type"]
        if column in conflict_details.columns
    ]

    if columns_to_show:
        print(
            conflict_details[columns_to_show]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print("Could not retrieve event details.")

    output_path = (
        PROJECT_ROOT / "data" / "processed" / "conflicting_records.csv"
    )

    conflict_details.to_csv(output_path, index=False)
    print("\nDetailed records saved to:", output_path)


if __name__ == "__main__":
    main()
