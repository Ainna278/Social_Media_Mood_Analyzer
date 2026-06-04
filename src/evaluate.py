"""
evaluate.py
Evaluation metrics for Social Media Mood Analyzer.
Computes accuracy, precision, recall, and F1-score.
"""

import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from pipeline_utils import MOOD_LABELS

os.makedirs("results/reports", exist_ok=True)


def evaluate(model, vectorizer, X_test, y_test):
    """
    Run full evaluation on the test set.
    Prints and saves accuracy, precision, recall, F1-score.
    """
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy  * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall    * 100:.2f}%")
    print(f"  F1-Score  : {f1        * 100:.2f}%")
    print()

    report = classification_report(
        y_test,
        y_pred,
        labels=MOOD_LABELS,
        target_names=MOOD_LABELS,
        zero_division=0
    )
    print("  Per-Class Report:")
    print(report)

    # Save report to file
    report_path = "results/reports/classification_report.txt"
    with open(report_path, "w") as f:
        f.write("SOCIAL MEDIA MOOD ANALYZER - EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy  : {accuracy  * 100:.2f}%\n")
        f.write(f"Precision : {precision * 100:.2f}%\n")
        f.write(f"Recall    : {recall    * 100:.2f}%\n")
        f.write(f"F1-Score  : {f1        * 100:.2f}%\n\n")
        f.write(report)
    print(f"  Report saved to: {report_path}")

    # Save summary CSV
    summary = pd.DataFrame([{
        "Accuracy":  round(accuracy,  4),
        "Precision": round(precision, 4),
        "Recall":    round(recall,    4),
        "F1-Score":  round(f1,        4),
    }])
    summary.to_csv("results/reports/evaluation_summary.csv", index=False)

    return y_pred
