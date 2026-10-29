
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMBINED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"
)

MODEL_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_model_data.csv"
)


def main():
    df = pd.read_csv(MODEL_PATH)
    combined = pd.read_csv(COMBINED_PATH)

    print("--- MISSING DATA INVESTIGATION ---")

    # 1. Investigate stream-order features.
    stream_columns = [
        column for column in df.columns
        if any(
            term in column.lower()
            for term in [
                "stream", "bifurcation", "order", "length ratio"
            ]
        )
    ]

    print("\n1. Missing values in stream-related features:")
    for column in stream_columns:
        count = int(df[column].isna().sum())
        if count > 0:
            print(f"{column}: {count} missing")

    print("\nRows with missing stream-related values:")
    if stream_columns:
        missing_stream = df[stream_columns].isna().any(axis=1)
        print(int(missing_stream.sum()))

        print("\nMissing stream values by gauge (top 15):")
        if "GaugeID" in combined.columns:
            gauge_missing = combined.loc[
                missing_stream, "GaugeID"
            ].value_counts()
            print(gauge_missing.head(15).to_string())

    # 2. Inspect inconsistent Basin spellings.
    print("\n2. Basin names after trimming and case normalization:")

    basin = df["Basin"].dropna().astype(str)
    basin_table = pd.DataFrame({
        "Original": basin,
        "Normalized": basin.map(
            lambda value: " ".join(value.split()).casefold()
        )
    })

    variants = basin_table.groupby("Normalized")["Original"].agg(
        lambda values: sorted(set(values))
    )

    for normalized, names in variants.items():
        if len(names) > 1:
            print(f"\nPossible duplicate names: {names}")

    print("\nBasin values with surrounding or repeated spaces:")
    for value in sorted(basin.unique()):
        if value != " ".join(value.split()):
            print(repr(value))

    # 3. Inspect records with missing Basin.
    print("\n3. Records with missing Basin:")
    missing_basin = df["Basin"].isna()

    if "GaugeID" in combined.columns:
        print(
            combined.loc[
                missing_basin,
                ["EventID", "GaugeID", "Basin", "State"]
            ].drop_duplicates().head(20).to_string(index=False)
        )

    # 4. Check whether Basin is missing in the combined source data.
    print("\n4. Missing Basin in combined dataset:")
    if "Basin" in combined.columns:
        print(int(combined["Basin"].isna().sum()))

    # 5. Summarize missing values by row.
    print("\n5. Number of missing cells per row in model dataset:")
    print(df.isna().sum(axis=1).describe().round(2))

    print("\nInvestigation complete. No data has been modified.")


if __name__ == "__main__":
    main()
