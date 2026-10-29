
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    rainfall_columns = [
        f"T{i}d" for i in range(1, 11)
    ]

    print("--- Rainfall dataset check ---")
    print("Dataset shape:", df.shape)

    print("\n--- Rainfall summary ---")
    print(df[rainfall_columns].describe().round(2))

    print("\n--- Percentage of zero rainfall values ---")
    for column in rainfall_columns:
        zero_percentage = (df[column] == 0).mean() * 100
        print(f"{column}: {zero_percentage:.2f}%")

    print("\n--- Rainfall variation ---")
    for column in rainfall_columns:
        print(
            f"{column}: "
            f"{df[column].nunique()} unique values"
        )

    print("\n--- Sample events with all rainfall values equal to zero ---")
    all_zero = (df[rainfall_columns] == 0).all(axis=1)

    print("Number of events:", int(all_zero.sum()))

    print(
        df.loc[
            all_zero,
            ["EventID", "Start Date", "Flood Type"] + rainfall_columns
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
