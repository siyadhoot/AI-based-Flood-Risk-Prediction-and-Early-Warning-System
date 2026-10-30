"""Evaluate tuned models once on the baseline held-out test set."""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from .tune import TUNED_MODEL_DIR, TUNED_OUTPUT_DIR
from .train import OUTPUT_DIR


def evaluate_tuned():
    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv", low_memory=False)
    y_true = pd.read_csv(OUTPUT_DIR / "y_test.csv").iloc[:, 0].astype(str)
    labels = sorted(y_true.unique())
    predictions = {"actual": y_true}
    scores_by_model, classes_by_model = {}, {}
    reports, matrices, rows = {}, {}, []
    timings = pd.read_csv(TUNED_OUTPUT_DIR / "training_times.csv").set_index("model").training_seconds
    for model_path in sorted(TUNED_MODEL_DIR.glob("*.joblib")):
        name = model_path.stem
        pipeline = joblib.load(model_path)
        y_pred = pd.Series(pipeline.predict(X_test)).astype(str)
        predictions[f"{name}__predicted"] = y_pred
        if hasattr(pipeline, "predict_proba"):
            scores_by_model[name] = pipeline.predict_proba(X_test).tolist()
            classes_by_model[name] = [str(x) for x in pipeline.named_steps["classifier"].classes_]
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        reports[name] = report
        matrices[name] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        row = {
            "model": name, "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "training_seconds": float(timings[name]),
        }
        if name in scores_by_model and len(labels) == 2:
            class_names = classes_by_model[name]
            positive = class_names.index("Severe Flood")
            values = np.asarray(scores_by_model[name])[:, positive]
            truth_binary = (y_true == "Severe Flood").astype(int)
            row["roc_auc"] = roc_auc_score(truth_binary, values)
            row["average_precision"] = average_precision_score(truth_binary, values)
        else:
            row["roc_auc"] = row["average_precision"] = np.nan
        for label in labels:
            for metric in ("precision", "recall", "f1-score", "support"):
                row[f"{label}_{metric}"] = report[label][metric]
        rows.append(row)
    predictions_df = pd.DataFrame(predictions)
    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)

    # Append the previously saved ANN test result; it used this identical test partition.
    ann_metrics_path = OUTPUT_DIR.parent / "ann" / "test_metrics.json"
    if ann_metrics_path.exists():
        ann = json.loads(ann_metrics_path.read_text(encoding="utf-8"))
        results = pd.concat([results, pd.DataFrame([ann])], ignore_index=True).sort_values("f1_macro", ascending=False)
        reports["ann"] = json.loads((OUTPUT_DIR.parent / "ann" / "classification_report.json").read_text(encoding="utf-8"))
        matrices["ann"] = json.loads((OUTPUT_DIR.parent / "ann" / "confusion_matrix.json").read_text(encoding="utf-8"))["matrix"]
        ann_probabilities = np.load(OUTPUT_DIR.parent / "ann" / "test_predictions.npz")["probabilities"]
        scores_by_model["ann"] = np.column_stack([1 - ann_probabilities, ann_probabilities]).tolist()
        classes_by_model["ann"] = ["Flood", "Severe Flood"]
        predictions_df["ann__predicted"] = np.where(ann_probabilities >= 0.5, "Severe Flood", "Flood")
    results["model_type"] = np.where(results.model.eq("ann"), "Deep Learning", "Traditional ML")
    results.to_csv(TUNED_OUTPUT_DIR / "model_comparison.csv", index=False)
    predictions_df.to_csv(TUNED_OUTPUT_DIR / "test_predictions.csv", index=False)
    (TUNED_OUTPUT_DIR / "test_scores.json").write_text(json.dumps({
        "classes_by_model": classes_by_model, "probabilities_by_model": scores_by_model,
        "note": "Scores are not calibrated real-world probabilities.",
    }), encoding="utf-8")
    (TUNED_OUTPUT_DIR / "classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (TUNED_OUTPUT_DIR / "confusion_matrices.json").write_text(json.dumps({"labels": labels, "matrices": matrices}, indent=2), encoding="utf-8")
    _write_comparison_charts(results)
    print(results[["model", "accuracy", "f1_macro", "Severe Flood_recall"]].to_string(index=False))


def _write_comparison_charts(results):
    for metric, title in [("accuracy", "Accuracy"), ("precision_macro", "Macro precision"),
                          ("recall_macro", "Macro recall"), ("f1_macro", "Macro F1-score"),
                          ("Severe Flood_recall", "Severe Flood recall")]:
        values = pd.to_numeric(results[metric], errors="coerce")
        colors = ["#16a3a1" if name == "ann" else "#3478b9" for name in results.model]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(results.model, values, color=colors)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=8)
        ax.set(title=title, ylabel="Score", ylim=(0, min(1.12, max(0.8, float(values.max()) + 0.12))))
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout(); fig.savefig(TUNED_OUTPUT_DIR / f"comparison_{metric.lower().replace(' ', '_')}.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    evaluate_tuned()
