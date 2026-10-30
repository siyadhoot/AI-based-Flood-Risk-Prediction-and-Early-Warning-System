import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.ann.inference import predict_ann


class FakeModel:
    def predict(self, values, verbose=0):
        assert values.shape == (1, 3)
        return np.array([[0.8]])


def test_ann_preprocessing_and_prediction_keep_feature_order():
    train = pd.DataFrame({"rain": [1.0, 3.0], "basin": ["A", "B"]})
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["rain"]),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["basin"]),
    ], sparse_threshold=0.0)
    preprocessor.fit(train)
    row = pd.DataFrame({"rain": [2.0], "basin": ["unseen"]})
    label, probability = predict_ann(FakeModel(), preprocessor, row, ["rain", "basin"])
    assert label == "Severe Flood"
    assert probability == pytest.approx(0.8)


def test_ann_inference_rejects_wrong_feature_order():
    features = pd.DataFrame({"rain": [1.0], "basin": ["A"]})
    with pytest.raises(ValueError, match="feature order"):
        predict_ann(FakeModel(), object(), features, ["basin", "rain"])


def test_keras_checkpoint_loads_and_predicts(tmp_path):
    model_path = tmp_path / "checkpoint.keras"
    model = tf.keras.Sequential([tf.keras.layers.Input(shape=(3,)), tf.keras.layers.Dense(1, activation="sigmoid")])
    model.save(model_path)
    loaded = tf.keras.models.load_model(model_path)
    train = pd.DataFrame({"rain": [1.0, 3.0], "basin": ["A", "B"]})
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["rain"]),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ["basin"]),
    ], sparse_threshold=0.0).fit(train)
    label, probability = predict_ann(loaded, preprocessor, pd.DataFrame({"rain": [2.0], "basin": ["A"]}), ["rain", "basin"])
    assert label in {"Flood", "Severe Flood"}
    assert 0 <= probability <= 1
