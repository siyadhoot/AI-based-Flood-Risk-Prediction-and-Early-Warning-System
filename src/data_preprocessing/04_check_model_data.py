
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "flood_model_data.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    print("Dataset shape:", df.shape)

    print("\n--- Target distribution ---")
    print(df["Flood Type"].value_counts())

    print("\n--- Missing values ---")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.to_string())
        print("\nTotal missing cells:", int(missing.sum()))

    print("\n--- Data types ---")
    print(df.dtypes.value_counts())

    print("\n--- Categorical columns ---")
    categorical = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    print(categorical)

    print("\n--- Numerical summary ---")
    print(df.select_dtypes(include="number").describe().round(2))

    print("\n--- Duplicate rows ---")
    print("Duplicate rows:", df.duplicated().sum())

    print("\n--- Model data check complete ---")


if __name__ == "__main__":
    main()
