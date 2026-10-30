"""Select traditional model hyperparameters by CV on the saved training split.

Run: python -m src.model_evaluation.tune
The held-out test partition is not read by this module.
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import loguniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .train import OUTPUT_DIR, MODEL_DIR, RANDOM_STATE, TARGET, make_preprocessor

TUNED_MODEL_DIR = MODEL_DIR / "tuned"
TUNED_OUTPUT_DIR = OUTPUT_DIR / "tuned"
CV_FOLDS = 3
N_ITER = 8


def tuning_specs():
    return {
        "gradient_boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False, {
            "classifier__n_estimators": [50, 80, 120, 160],
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "classifier__max_depth": [1, 2, 3],
            "classifier__min_samples_leaf": [1, 3, 6, 10],
            "classifier__subsample": [0.75, 0.9, 1.0],
        }),
        "random_forest": (RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1), False, {
            "classifier__n_estimators": [120, 200, 300],
            "classifier__max_depth": [None, 8, 12, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", 0.5, 0.8],
            "classifier__class_weight": [None, "balanced", "balanced_subsample"],
        }),
        "decision_tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), False, {
            "classifier__max_depth": [None, 4, 6, 8, 12, 16],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8, 12],
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__class_weight": [None, "balanced"],
        }),
        "logistic_regression": (LogisticRegression(max_iter=3000, random_state=RANDOM_STATE), True, {
            "classifier__C": loguniform(1e-3, 1e2),
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ["lbfgs"],
        }),
        "svc_rbf": (SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), True, {
            "classifier__C": loguniform(1e-2, 1e2),
            "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
            "classifier__class_weight": [None, "balanced"],
        }),
        "knn": (KNeighborsClassifier(), True, {
            "classifier__n_neighbors": [3, 5, 7, 9, 13, 17, 21],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        }),
        "gaussian_nb": (GaussianNB(), True, {
            "classifier__var_smoothing": loguniform(1e-12, 1e-7),
        }),
        "dummy": (DummyClassifier(random_state=RANDOM_STATE), False, {
            "classifier__strategy": ["most_frequent", "prior", "stratified", "uniform"],
        }),
    }


def tune_models():
    TUNED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv", low_memory=False)
    y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv").iloc[:, 0].astype(str)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for name, (estimator, scale_numeric, search_space) in tuning_specs().items():
        pipeline = Pipeline([
            ("preprocess", make_preprocessor(X_train, scale_numeric)),
            ("classifier", estimator),
        ])
        search = RandomizedSearchCV(
            pipeline, search_space, n_iter=min(N_ITER, 4 if name == "dummy" else N_ITER),
            scoring="f1_macro", cv=cv, n_jobs=1, refit=True,
            random_state=RANDOM_STATE, error_score="raise", return_train_score=True,
            verbose=1,
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        joblib.dump(search.best_estimator_, TUNED_MODEL_DIR / f"{name}.joblib")
        records.append({
            "model": name,
            "cv_macro_f1": float(search.best_score_),
            "cv_train_macro_f1": float(search.cv_results_["mean_train_score"][search.best_index_]),
            "search_and_refit_seconds": elapsed,
            "best_params": search.best_params_,
            "candidates_evaluated": len(search.cv_results_["params"]),
        })
        print(f"{name}: CV macro F1={search.best_score_:.4f}; params={search.best_params_}")
    (TUNED_OUTPUT_DIR / "hyperparameter_search.json").write_text(json.dumps({
        "selection_metric": "macro F1",
        "cv": f"{CV_FOLDS}-fold shuffled StratifiedKFold",
        "random_state": RANDOM_STATE,
        "search_method": "RandomizedSearchCV; no held-out test data used for selection",
        "iterations_per_model": N_ITER,
        "training_rows": len(X_train),
        "models": records,
    }, indent=2), encoding="utf-8")
    (TUNED_OUTPUT_DIR / "training_times.csv").write_text(
        pd.DataFrame([{"model": x["model"], "training_seconds": x["search_and_refit_seconds"]} for x in records]).to_csv(index=False),
        encoding="utf-8",
    )
    print(f"Saved tuned pipelines to {TUNED_MODEL_DIR}; test partition was not accessed.")


if __name__ == "__main__":
    tune_models()
