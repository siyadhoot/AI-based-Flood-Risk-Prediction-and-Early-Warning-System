
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def inspect_dataset(filename, dataset_name):
    """Load a CSV and display basic information."""
    file_path = RAW_DIR / filename
    df = pd.read_csv(file_path)

    print(f"\n{'=' * 50}")
    print(f"DATASET: {dataset_name}")
    print("=" * 50)
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    return df


def main():
    flood_df = inspect_dataset(
        "floodevents_indofloods.csv",
        "Flood Events"
    )

    precipitation_df = inspect_dataset(
        "precipitation_variables_indofloods.csv",
        "Precipitation"
    )

    metadata_df = inspect_dataset(
        "metadata_indofloods.csv",
        "Metadata"
    )

    catchment_df = inspect_dataset(
        "catchment_characteristics_indofloods.csv",
        "Catchment Characteristics"
    )

    print("\n" + "=" * 50)
    print("CHECKING DATASET CONNECTIONS")
    print("=" * 50)

    flood_df["GaugeID"] = (
        flood_df["EventID"].str.rsplit("-", n=1).str[0]
    )

    print("\nSample extracted gauge IDs:")
    print(flood_df[["EventID", "GaugeID"]].head())

    print("\nFlood events matching precipitation:")
    print(
        flood_df["EventID"].isin(precipitation_df["EventID"]).sum(),
        "out of", len(flood_df)
    )

    print("\nFlood gauges matching metadata:")
    print(
        flood_df["GaugeID"].isin(metadata_df["GaugeID"]).sum(),
        "out of", len(flood_df)
    )

    print("\nFlood gauges matching catchment characteristics:")
    print(
        flood_df["GaugeID"].isin(catchment_df["GaugeID"]).sum(),
        "out of", len(flood_df)
    )


    # Combine all four datasets

    merged_df = flood_df.merge(
        precipitation_df,
        on="EventID",
        how="left",
        validate="one_to_one"
    )

    merged_df = merged_df.merge(
        metadata_df,
        on="GaugeID",
        how="left",
        suffixes=("", "_metadata")
    )

    merged_df = merged_df.merge(
        catchment_df,
        on="GaugeID",
        how="left",
        suffixes=("", "_catchment")
    )

    # Create the processed data folder if needed
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save the combined dataset
    output_path = PROCESSED_DIR / "flood_combined.csv"
    merged_df.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print("COMBINED DATASET CREATED")
    print("=" * 50)
    print("Shape:", merged_df.shape)
    print("Missing values:", merged_df.isnull().sum().sum())
    print("Saved to:", output_path)
    print("\nFlood Type distribution:")
    print(merged_df["Flood Type"].value_counts())



if __name__ == "__main__":
    main()
