"""Evaluate saved ANN and consolidate it with existing baseline test metrics."""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve, precision_recall_curve)

from .train import EVAL_DIR, OUTPUT_DIR, CLASSES


def evaluate_ann():
    import tensorflow as tf
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run = json.loads((OUTPUT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    y_true = pd.read_csv(EVAL_DIR / "y_test.csv").iloc[:, 0].astype(str)
    probabilities = np.load(OUTPUT_DIR / "test_predictions.npz")["probabilities"]
    y_pred = np.where(probabilities >= 0.5, "Severe Flood", "Flood")
    binary = (y_true == "Severe Flood").astype(int)
    report = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    row = {"model": "ann", "accuracy": accuracy_score(y_true, y_pred),
           "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
           "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
           "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
           "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
           "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
           "roc_auc": roc_auc_score(binary, probabilities), "average_precision": average_precision_score(binary, probabilities),
           "training_seconds": run["training_seconds"], "inference_seconds": run["inference_seconds"]}
    for label in CLASSES:
        for metric in ("precision", "recall", "f1-score", "support"):
            row[f"{label}_{metric}"] = report[label][metric]
    (OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "confusion_matrix.json").write_text(json.dumps({"labels": CLASSES, "matrix": matrix}, indent=2), encoding="utf-8")

    history = pd.read_csv(OUTPUT_DIR / "training_history.csv")
    for metric, title, filename in [("accuracy", "Accuracy", "training_accuracy.png"), ("loss", "Loss", "training_loss.png")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(history[metric], label="Training")
        ax.plot(history[f"val_{metric}"], label="Validation")
        ax.set(title=f"ANN training and validation {title.lower()}", xlabel="Epoch", ylabel=title)
        ax.legend(); fig.tight_layout(); fig.savefig(OUTPUT_DIR / filename, dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(2), xticklabels=CLASSES, yticks=range(2), yticklabels=CLASSES, xlabel="Predicted", ylabel="Actual", title="ANN confusion matrix")
    for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, value, ha="center", va="center")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); fpr, tpr, _ = roc_curve(binary, probabilities); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ANN ROC curve"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4)); precision, recall, _ = precision_recall_curve(binary, probabilities); ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="ANN precision-recall curve"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=160); plt.close(fig)

    baseline = pd.read_csv(EVAL_DIR / "model_comparison.csv")
    baseline = baseline.rename(columns={"precision_macro": "precision_macro", "recall_macro": "recall_macro"})
    comparison = pd.concat([baseline, pd.DataFrame([row])], ignore_index=True).sort_values("f1_macro", ascending=False)
    comparison["model_type"] = np.where(comparison.model.eq("ann"), "Deep Learning", "Traditional ML")
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    comparison.to_csv(EVAL_DIR / "model_comparison_with_ann.csv", index=False)
    for metric, title, filename in [
        ("accuracy", "Accuracy", "comparison_accuracy.png"),
        ("precision_macro", "Macro precision", "comparison_macro_precision.png"),
        ("recall_macro", "Macro recall", "comparison_macro_recall.png"),
        ("f1_macro", "Macro F1-score", "comparison_macro_f1.png"),
        ("Severe Flood_recall", "Severe Flood recall", "comparison_severe_recall.png"),
    ]:
        values = pd.to_numeric(comparison[metric], errors="coerce")
        colors = ["#16a3a1" if name == "ann" else "#3478b9" for name in comparison.model]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        bars = ax.bar(comparison.model, values, color=colors)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=8)
        ax.set(title=title, ylabel="Score", ylim=(0, min(1.12, max(0.8, float(values.max()) + 0.12))))
        ax.tick_params(axis="x", rotation=35)
        ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#3478b9", label="Traditional ML"),
                           plt.Rectangle((0, 0), 1, 1, color="#16a3a1", label="Deep Learning")])
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / filename, dpi=160); plt.close(fig)
    merged_reports = json.loads((EVAL_DIR / "classification_reports.json").read_text(encoding="utf-8"))
    merged_reports["ann"] = report
    (OUTPUT_DIR / "classification_reports.json").write_text(json.dumps(merged_reports, indent=2), encoding="utf-8")
    merged_matrices = json.loads((EVAL_DIR / "confusion_matrices.json").read_text(encoding="utf-8"))
    merged_matrices["matrices"]["ann"] = matrix
    (OUTPUT_DIR / "confusion_matrices.json").write_text(json.dumps(merged_matrices, indent=2), encoding="utf-8")
    print(comparison[["model", "accuracy", "f1_macro", "Severe Flood_recall"]].to_string(index=False))


if __name__ == "__main__":
    evaluate_ann()
