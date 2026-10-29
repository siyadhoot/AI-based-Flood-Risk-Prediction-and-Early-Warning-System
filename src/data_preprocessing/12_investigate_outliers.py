
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT / "data" / "processed" / "flood_basic_cleaned.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    columns = [
        "Event Duration (days)",
        "Peak Flood Level (m)",
        "Num Peak FL",
        "Peak Discharge Q (cumec)",
        "Flood Volume (cumec)",
        "Time to Peak (days)",
        "Recession Time (day)",
    ]

    print("--- NUMERICAL OUTLIER INVESTIGATION ---")

    for column in columns:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = values[(values < lower) | (values > upper)]

        print(f"\n{column}")
        print("Valid values:", len(values))
        print("Minimum:", values.min())
        print("Median:", values.median())
        print("Maximum:", values.max())
        print("Potential outliers:", len(outliers))
        print("IQR lower bound:", round(lower, 3))
        print("IQR upper bound:", round(upper, 3))

        if not outliers.empty:
            print("Largest unusual values:")
            print(outliers.sort_values(ascending=False).head(5).to_list())


if __name__ == "__main__":
    main()
