
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    print("=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)

    print("\n1. Dataset shape:")
    print(df.shape)

    print("\n2. Duplicate event IDs:")
    print(df["EventID"].duplicated().sum())

    print("\n3. Duplicate rows:")
    print(df.duplicated().sum())

    print("\n4. Flood type distribution:")
    print(df["Flood Type"].value_counts())

    print("\n5. Missing values in columns:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    print(missing.to_string())

    print("\n6. Missing values by percentage:")
    missing_percent = (df.isnull().mean() * 100)
    missing_percent = missing_percent[
        missing_percent > 0
    ].sort_values(ascending=False)
    print(missing_percent.round(2).to_string())

    print("\n7. Data types:")
    print(df.dtypes.value_counts())

    print("\n8. Numeric summary:")
    print(df.select_dtypes(include="number").describe().T)

    print("\nDATA QUALITY CHECK COMPLETE")


if __name__ == "__main__":
    main()
