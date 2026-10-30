"""Train a small Keras ANN on the existing baseline train/test partition.

Run from the repository root: python -m src.ann.train
"""
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "ann"
EVAL_DIR = ROOT / "outputs" / "model_evaluation"
TARGET = "Flood Type"
SEED = 42
CLASSES = ["Flood", "Severe Flood"]


def train_ann():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required. Install project dependencies with: pip install -r requirements.txt") from exc

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(EVAL_DIR / "X_train.csv", low_memory=False)
    y_train = pd.read_csv(EVAL_DIR / "y_train.csv").iloc[:, 0].astype(str)
    X_test = pd.read_csv(EVAL_DIR / "X_test.csv", low_memory=False)
    y_test = pd.read_csv(EVAL_DIR / "y_test.csv").iloc[:, 0].astype(str)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Saved train and test feature orders differ")
    if set(y_train.unique()) != set(CLASSES) or set(y_test.unique()) != set(CLASSES):
        raise ValueError(f"Expected target classes {CLASSES}")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train,
    )
    numeric = X_fit.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in X_fit.columns if column not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop", sparse_threshold=0.0)
    # The preprocessing fit sees only the training fold; validation and test are transform-only.
    x_fit = np.asarray(preprocessor.fit_transform(X_fit), dtype=np.float32)
    x_val = np.asarray(preprocessor.transform(X_val), dtype=np.float32)
    x_test = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    y_fit_bin = (y_fit == "Severe Flood").astype("float32").to_numpy()
    y_val_bin = (y_val == "Severe Flood").astype("float32").to_numpy()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x_fit.shape[1],)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    counts = y_fit.value_counts()
    class_weight = {0: len(y_fit) / (2 * counts["Flood"]), 1: len(y_fit) / (2 * counts["Severe Flood"])}
    checkpoint = OUTPUT_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
    ]
    start = time.perf_counter()
    history = model.fit(x_fit, y_fit_bin, validation_data=(x_val, y_val_bin), epochs=200,
                        batch_size=32, class_weight=class_weight, callbacks=callbacks, verbose=2)
    training_seconds = time.perf_counter() - start
    best = tf.keras.models.load_model(checkpoint)
    probabilities = best.predict(x_test, verbose=0).reshape(-1)
    inference_start = time.perf_counter()
    best.predict(x_test, verbose=0)
    inference_seconds = time.perf_counter() - inference_start
    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.joblib")
    joblib.dump(preprocessor.named_transformers_["numeric"].named_steps["scaler"], OUTPUT_DIR / "scaler.joblib")
    joblib.dump({"feature_names": X_train.columns.tolist(), "classes": CLASSES, "positive_class": "Severe Flood"}, OUTPUT_DIR / "metadata.joblib")
    pd.DataFrame(history.history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    np.savez_compressed(OUTPUT_DIR / "test_predictions.npz", probabilities=probabilities)
    (OUTPUT_DIR / "run_metadata.json").write_text(json.dumps({
        "seed": SEED, "train_rows": len(X_fit), "validation_rows": len(X_val), "test_rows": len(X_test),
        "epochs_completed": len(history.epoch), "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "training_seconds": training_seconds, "inference_seconds": inference_seconds,
        "class_weight": {"Flood": class_weight[0], "Severe Flood": class_weight[1]},
        "split_protocol": "Existing stratified random 80/20 baseline split; 20% of training partition held for validation; no GaugeID grouping in baseline.",
        "architecture": [128, 128, 64, 32, 1], "hidden_activation": "relu",
        "output_activation": "sigmoid", "batch_size": 32,
        "learning_rate": 0.001, "patience": 15,
    }, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint; completed {len(history.epoch)} epochs (best epoch {int(np.argmin(history.history['val_loss']) + 1)}).")


if __name__ == "__main__":
    train_ann()
