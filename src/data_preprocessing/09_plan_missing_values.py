
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_model_data.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    print("--- MISSING VALUE AND FEATURE REVIEW ---")
    print("Dataset shape:", df.shape)

    print("\n1. Columns with missing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    for column, count in missing.items():
        percentage = count / len(df) * 100
        print(f"{column}: {count} missing ({percentage:.2f}%)")

    print("\nTotal missing cells:", int(df.isna().sum().sum()))

    print("\n2. Categorical feature values:")
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for column in categorical_columns:
        if column == "Flood Type":
            continue

        print(f"\n{column}")
        print("Unique values:", df[column].nunique(dropna=True))
        print("Missing values:", int(df[column].isna().sum()))
        print(df[column].value_counts(dropna=False).head(10).to_string())

    print("\n3. Target distribution:")
    print(df["Flood Type"].value_counts())

    print("\nReview complete. No data has been modified.")


if __name__ == "__main__":
    main()
