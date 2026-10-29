"""Calculate comparison metrics and create plots from test.py outputs.

Run after train.py and test.py: python -m src.model_evaluation.evaluate
"""
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from .train import OUTPUT_DIR, model_specs


def evaluate_models():
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv", low_memory=False)
    score_data = json.loads((OUTPUT_DIR / "test_scores.json").read_text(encoding="utf-8"))
    train_times = pd.read_csv(OUTPUT_DIR / "training_times.csv").set_index("model")["training_seconds"]
    y_true = predictions["actual"].astype(str)
    labels = sorted(y_true.unique())
    rows, reports, matrices = [], {}, {}
    for name, _, _ in model_specs():
        y_pred = predictions[f"{name}__predicted"].astype(str)
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        reports[name] = report
        matrices[name] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
        scores = score_data["probabilities_by_model"].get(name)
        auc = average_precision = np.nan
        try:
            if scores is not None and len(labels) == 2:
                model_classes = score_data["classes_by_model"][name]
                positive_index = model_classes.index(labels[1])
                positive_scores = np.asarray(scores)[:, positive_index]
                binary_truth = (y_true == labels[1]).astype(int)
                auc = roc_auc_score(binary_truth, positive_scores)
                average_precision = average_precision_score(binary_truth, positive_scores)
        except (ValueError, KeyError, IndexError):
            pass
        row = {
            "model": name, "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "roc_auc": auc, "average_precision": average_precision,
            "training_seconds": train_times[name],
        }
        for label in labels:
            for metric in ("precision", "recall", "f1-score", "support"):
                row[f"{label}_{metric}"] = report[label][metric]
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "confusion_matrices.json").write_text(json.dumps({"labels": labels, "matrices": matrices}, indent=2), encoding="utf-8")
    _make_plots(results, labels, matrices, predictions, score_data)
    print("Saved evaluation metrics and plots to outputs/model_evaluation/")
    print(results[["model", "accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]].to_string(index=False))


def _make_plots(results, labels, matrices, predictions, score_data):
    def save(filename):
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close()

    ax = results.plot.bar(x="model", y="f1_macro", legend=False, figsize=(10, 5))
    ax.set(title="Macro F1 by model", ylabel="Macro F1", xlabel="Model")
    ax.tick_params(axis="x", rotation=35); save("macro_f1.png")
    ax = results.set_index("model")[["accuracy", "balanced_accuracy"]].plot.bar(figsize=(10, 5))
    ax.set(title="Accuracy and balanced accuracy", ylabel="Score", xlabel="Model")
    ax.tick_params(axis="x", rotation=35); save("accuracy_balanced_accuracy.png")
    recalls = [f"{label}_recall" for label in labels]
    ax = results.set_index("model")[recalls].plot.bar(figsize=(10, 5))
    ax.set(title="Per-class recall", ylabel="Recall", xlabel="Model")
    ax.tick_params(axis="x", rotation=35); save("per_class_recall.png")

    selected = results.head(3)["model"].tolist()
    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4))
    for ax, name in zip(np.atleast_1d(axes), selected):
        ax.imshow(matrices[name], cmap="Blues"); ax.set_title(name)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_xticks(range(len(labels)), labels, rotation=30); ax.set_yticks(range(len(labels)), labels)
        for (i, j), value in np.ndenumerate(matrices[name]): ax.text(j, i, value, ha="center", va="center")
    save("leading_confusion_matrices.png")

    if len(labels) == 2:
        truth = (predictions.actual.astype(str) == labels[1]).astype(int)
        for name, all_scores in score_data["probabilities_by_model"].items():
            classes = score_data["classes_by_model"][name]
            values = np.asarray(all_scores)[:, classes.index(labels[1])]
            fpr, tpr, _ = roc_curve(truth, values); plt.plot(fpr, tpr, label=name)
        plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title("ROC curves"); plt.legend(fontsize=7); save("roc_curves.png")
        for name, all_scores in score_data["probabilities_by_model"].items():
            classes = score_data["classes_by_model"][name]
            values = np.asarray(all_scores)[:, classes.index(labels[1])]
            precision, recall, _ = precision_recall_curve(truth, values)
            plt.plot(recall, precision, label=name)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curves")
        plt.legend(fontsize=7); save("precision_recall_curves.png")


if __name__ == "__main__":
    evaluate_models()
