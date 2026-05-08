"""
visualize.py
Visualizations for Social Media Mood Analyzer evaluation.
Generates confusion matrix and per-class metric bar chart.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

MOOD_LABELS = ["happy", "sad", "angry", "neutral"]
os.makedirs("results/charts", exist_ok=True)


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save confusion matrix (raw counts and normalised)."""

    cm = confusion_matrix(y_test, y_pred, labels=MOOD_LABELS)

    # Raw count
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix (Raw Counts)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Mood", fontsize=11)
    ax.set_ylabel("True Mood", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/charts/confusion_matrix_raw.png", dpi=150)
    plt.close()
    print("  Saved: results/charts/confusion_matrix_raw.png")

    # Normalised
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix (Normalised)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Mood", fontsize=11)
    ax.set_ylabel("True Mood", fontsize=11)
    plt.tight_layout()
    plt.savefig("results/charts/confusion_matrix_normalised.png", dpi=150)
    plt.close()
    print("  Saved: results/charts/confusion_matrix_normalised.png")


def plot_metrics_chart(y_test, y_pred):
    """Bar chart showing Precision, Recall, F1-Score per mood class."""

    report = classification_report(
        y_test, y_pred,
        target_names=MOOD_LABELS,
        output_dict=True,
        zero_division=0
    )

    precision = [report[c]["precision"] for c in MOOD_LABELS]
    recall    = [report[c]["recall"]    for c in MOOD_LABELS]
    f1        = [report[c]["f1-score"]  for c in MOOD_LABELS]

    x     = np.arange(len(MOOD_LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precision, width, label="Precision", color="#4c72b0")
    ax.bar(x,         recall,    width, label="Recall",    color="#55a868")
    ax.bar(x + width, f1,        width, label="F1-Score",  color="#c44e52")

    ax.set_title("Per-Class Evaluation Metrics", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mood Category", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MOOD_LABELS, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("results/charts/per_class_metrics.png", dpi=150)
    plt.close()
    print("  Saved: results/charts/per_class_metrics.png")


def plot_all(y_test, y_pred):
    """Run all visualizations."""
    print("\n  Generating charts...")
    plot_confusion_matrix(y_test, y_pred)
    plot_metrics_chart(y_test, y_pred)
