
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_combined.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_basic_cleaned.csv"
)


def normalize_text(value):
    """Trim extra spaces and standardize capitalization."""
    if pd.isna(value):
        return value

    value = " ".join(str(value).split())

    # Basin names are treated case-insensitively.
    return value.casefold()


def main():
    df = pd.read_csv(INPUT_PATH)

    print("--- BASIC DATA CLEANING ---")
    print("Original shape:", df.shape)

    # 1. Standardize whitespace and capitalization in text columns.
    text_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    for column in text_columns:
        df[column] = df[column].map(
            lambda value: (
                " ".join(value.split())
                if isinstance(value, str)
                else value
            )
        )

    print("\nWhitespace cleaned in text columns:", len(text_columns))

    # 2. Normalize Basin names consistently.
    if "Basin" in df.columns:
        df["Basin"] = df["Basin"].map(normalize_text)

    print("Basin names standardized.")

    # 3. Remove exact duplicate rows only.
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)

    print("\nExact duplicate rows removed:", removed)
    print("Final shape:", df.shape)

    # 4. Save a separate output file.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nMissing cells remaining:", int(df.isna().sum().sum()))
    print("Output saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
