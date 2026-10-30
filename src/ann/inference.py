"""Small, testable ANN inference helpers shared by the dashboard."""
import numpy as np


def predict_ann(model, preprocessor, features, feature_names):
    if list(features.columns) != list(feature_names):
        raise ValueError("Input feature order does not match the saved ANN metadata")
    transformed = np.asarray(preprocessor.transform(features), dtype=np.float32)
    probability = float(np.asarray(model.predict(transformed, verbose=0)).reshape(-1)[0])
    if not 0 <= probability <= 1:
        raise ValueError("ANN returned an invalid probability")
    return ("Severe Flood" if probability >= 0.5 else "Flood"), probability
