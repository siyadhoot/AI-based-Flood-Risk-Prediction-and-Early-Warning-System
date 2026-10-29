"""Generate held-out predictions from the fixed split and saved pipelines.

Run after train.py: python -m src.model_evaluation.test
"""
import json

import joblib
import numpy as np
import pandas as pd

from .train import OUTPUT_DIR, model_path, model_specs


def test_models():
    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv", low_memory=False)
    y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv", low_memory=False)
    actual = y_test.iloc[:, 0]
    prediction_data = {"actual": actual.astype(str)}
    score_data = {}
    classes_by_model = {}

    for name, _, _ in model_specs():
        pipeline = joblib.load(model_path(name))
        predicted = pipeline.predict(X_test)
        prediction_data[f"{name}__predicted"] = pd.Series(predicted).astype(str)
        if hasattr(pipeline, "predict_proba"):
            scores = pipeline.predict_proba(X_test)
            classes = pipeline.named_steps["classifier"].classes_
            classes_by_model[name] = [str(c) for c in classes]
            score_data[name] = np.asarray(scores).tolist()

    pd.DataFrame(prediction_data).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    (OUTPUT_DIR / "test_scores.json").write_text(json.dumps({
        "classes_by_model": classes_by_model, "probabilities_by_model": score_data,
        "note": "Classifier scores are not calibrated real-world flood probabilities.",
    }), encoding="utf-8")
    print(f"Saved held-out predictions for {len(model_specs())} models.")


if __name__ == "__main__":
    test_models()
