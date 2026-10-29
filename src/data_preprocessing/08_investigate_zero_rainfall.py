
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMBINED_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"
)

PRECIPITATION_PATH = (
    PROJECT_ROOT / "data" / "raw" / "precipitation_variables_indofloods.csv"
)


def main():
    df = pd.read_csv(COMBINED_PATH)
    precipitation_df = pd.read_csv(PRECIPITATION_PATH)

    rainfall_columns = [f"T{i}d" for i in range(1, 11)]

    # Identify events where every rainfall value is zero.
    all_zero = (df[rainfall_columns] == 0).all(axis=1)
    zero_df = df.loc[all_zero].copy()

    print("--- ZERO-RAINFALL INVESTIGATION ---")

    print("\n1. Total zero-rainfall events:")
    print(len(zero_df))

    print("\n2. Flood category distribution:")
    print(zero_df["Flood Type"].value_counts())

    print("\n3. Percentage within each flood category:")
    print(
        pd.crosstab(
            df["Flood Type"],
            all_zero,
            normalize="index"
        ).round(4) * 100
    )

    print("\n4. Gauges with the most zero-rainfall events:")
    print(
        zero_df["GaugeID"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\n5. Sample zero-rainfall events:")
    print(
        zero_df[
            ["EventID", "GaugeID", "Start Date", "Flood Type"]
            + rainfall_columns
        ]
        .head(15)
        .to_string(index=False)
    )

    # Check the original precipitation file.
    raw_all_zero = (
        precipitation_df[rainfall_columns] == 0
    ).all(axis=1)

    print("\n6. Original precipitation file:")
    print("Total precipitation records:", len(precipitation_df))
    print(
        "Records with all ten rainfall values zero:",
        int(raw_all_zero.sum())
    )

    print("\n7. Are all zero-rainfall events also in the raw file?")

    raw_zero_ids = set(
        precipitation_df.loc[raw_all_zero, "EventID"]
    )
    combined_zero_ids = set(zero_df["EventID"])

    print(
        "Zero-rainfall events found in raw file:",
        len(combined_zero_ids & raw_zero_ids)
    )
    print(
        "Zero-rainfall events not found among raw zero records:",
        len(combined_zero_ids - raw_zero_ids)
    )


if __name__ == "__main__":
    main()
