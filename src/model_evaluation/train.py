"""Create the one fixed split and fit/save the eight model pipelines.

Run from the repository root: python -m src.model_evaluation.train
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "flood_model_data.csv"
OUTPUT_DIR = ROOT / "outputs" / "model_evaluation"
MODEL_DIR = ROOT / "models"
TARGET = "Flood Type"
RANDOM_STATE = 42

EXCLUDED_COLUMNS = {
    TARGET, "EventID", "GaugeID", "Station", "River Name/ Tributory/ SubTributory",
    "Start Date", "End Date", "Start_date", "End_date", "Peak Flood Level (m)",
    "Peak FL Date", "Num Peak FL", "Peak Discharge Q (cumec)", "Peak Discharge Date",
    "Flood Volume (cumec)", "Event Duration (days)", "Time to Peak (days)",
    "Recession Time (day)", "Danger Level", "Warning Level", "Reliability",
    "Privacy", "Level_Entries", "Streamflow_Entries",
}
UNVERIFIED_TIMING_COLUMNS = {f"T{i}d" for i in range(1, 11)}


def load_model_data(path=DATA_PATH):
    df = pd.read_csv(path, low_memory=False)
    if TARGET not in df:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    if df[TARGET].isna().any():
        raise ValueError("Target contains missing labels")
    return df.drop_duplicates().copy()


def make_features(df):
    excluded = EXCLUDED_COLUMNS | UNVERIFIED_TIMING_COLUMNS
    X = df.drop(columns=[c for c in excluded if c in df], errors="ignore").copy()
    X = X.dropna(axis=1, how="all")
    date_like = [c for c in X if "date" in c.casefold()]
    X = X.drop(columns=date_like)
    excluded_used = sorted((set(df.columns) - set(X.columns)) - {TARGET})
    return X, df[TARGET].copy(), excluded_used


def make_preprocessor(X, scale_numeric):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ], remainder="drop", sparse_threshold=0.0)


def model_specs():
    return [
        ("dummy", DummyClassifier(strategy="most_frequent"), False),
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE), True),
        ("decision_tree", DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE), False),
        ("knn", KNeighborsClassifier(n_neighbors=5), True),
        ("gaussian_nb", GaussianNB(), True),
        ("svc_rbf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE), True),
        ("random_forest", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
        ("gradient_boosting", GradientBoostingClassifier(random_state=RANDOM_STATE), False),
    ]


def model_path(name):
    return MODEL_DIR / f"{name}.joblib"


def train_models(data_path=DATA_PATH):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_model_data(data_path)
    X, y, excluded = make_features(df)
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify,
    )

    # Persist the single test partition for test.py and evaluate.py. These files
    # are generated artifacts and do not alter the processed source dataset.
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_test.rename(TARGET).to_csv(OUTPUT_DIR / "y_test.csv", index=False)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    y_train.rename(TARGET).to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    pd.DataFrame([
        {"partition": label, "rows": len(labels), "class_counts": json.dumps(labels.value_counts().to_dict())}
        for label, labels in (("train", y_train), ("test", y_test))
    ]).to_csv(OUTPUT_DIR / "split_summary.csv", index=False)
    (OUTPUT_DIR / "feature_exclusions.json").write_text(json.dumps({
        "excluded_columns": excluded,
        "notes": {c: "Rainfall window timing is not documented; excluded conservatively." if c.startswith("T") and c.endswith("d") else "Target, identifier, date, target-rule input, or event outcome." for c in excluded},
    }, indent=2), encoding="utf-8")

    timings = []
    for name, estimator, scale_numeric in model_specs():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor(X_train, scale_numeric)),
            ("classifier", estimator),
        ])
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        joblib.dump(pipeline, model_path(name))
        timings.append({"model": name, "training_seconds": elapsed})

    pd.DataFrame(timings).to_csv(OUTPUT_DIR / "training_times.csv", index=False)
    metadata = {
        "target": TARGET, "classes": [str(v) for v in sorted(y.unique())],
        "feature_names": X.columns.tolist(), "excluded_columns": excluded,
        "random_state": RANDOM_STATE, "test_size": 0.2,
        "stratified": stratify is not None,
        "models": [name for name, _, _ in model_specs()],
        "duplicate_rows_removed_in_memory": "exact duplicates only",
    }
    (MODEL_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(timings)} fitted pipelines. Test partition: {len(y_test)} rows.")


if __name__ == "__main__":
    train_models()
