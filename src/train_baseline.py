import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from pipeline_utils import MOOD_LABELS, get_train_test_split, print_dataset_summary


BASELINE_RESULTS_DIR = "results/baseline"
BASELINE_REPORTS_DIR = os.path.join(BASELINE_RESULTS_DIR, "reports")
BASELINE_CHARTS_DIR = os.path.join(BASELINE_RESULTS_DIR, "charts")
BASELINE_MODEL_PATH = os.path.join(BASELINE_RESULTS_DIR, "baseline_model.pkl")
BASELINE_VECTORIZER_PATH = os.path.join(BASELINE_RESULTS_DIR, "baseline_vectorizer.pkl")

os.makedirs(BASELINE_REPORTS_DIR, exist_ok=True)
os.makedirs(BASELINE_CHARTS_DIR, exist_ok=True)


def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1-Score": macro_f1,
    }


def save_charts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=MOOD_LABELS)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS)
    plt.title("Baseline Confusion Matrix (Raw)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(BASELINE_CHARTS_DIR, "confusion_matrix_raw.png"))
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float") / row_sums

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=MOOD_LABELS,
        yticklabels=MOOD_LABELS,
    )
    plt.title("Baseline Confusion Matrix (Normalised)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(BASELINE_CHARTS_DIR, "confusion_matrix_normalised.png"))
    plt.close()

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=MOOD_LABELS, zero_division=0)
    metrics_df = pd.DataFrame({"Precision": p, "Recall": r, "F1-Score": f1}, index=MOOD_LABELS)

    metrics_df.plot(kind="bar", figsize=(8, 5))
    plt.title("Baseline Per-Class Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(BASELINE_CHARTS_DIR, "per_class_metrics.png"))
    plt.close()


print("--- Step 1: Loading shared processed dataset and split ---")
train_df, test_df, stats = get_train_test_split()
print_dataset_summary(stats, train_df, test_df)

X_train = train_df["clean_text"]
y_train = train_df["mood"]
X_test = test_df["clean_text"]
y_test = test_df["mood"]

print("\n--- Step 2: Majority-class benchmark ---")
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_metrics = evaluate_predictions(y_test, dummy_pred)
for key in ["Accuracy", "Precision", "Recall", "F1-Score", "Macro F1-Score"]:
    print(f"{key}: {dummy_metrics[key]:.4f}")

print("\n--- Step 3: Baseline TF-IDF feature extraction ---")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 1))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\n--- Step 4: Training baseline Logistic Regression ---")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

joblib.dump(model, BASELINE_MODEL_PATH)
joblib.dump(vectorizer, BASELINE_VECTORIZER_PATH)

print(f"Baseline model saved to: {BASELINE_MODEL_PATH}")
print(f"Baseline vectorizer saved to: {BASELINE_VECTORIZER_PATH}")

print("\n--- Step 5: Evaluating baseline model ---")
metrics = evaluate_predictions(y_test, y_pred)

for key in ["Accuracy", "Precision", "Recall", "F1-Score", "Macro F1-Score"]:
    print(f"{key}: {metrics[key]:.4f}")

report = classification_report(y_test, y_pred, labels=MOOD_LABELS, zero_division=0)

with open(os.path.join(BASELINE_REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

summary_df = pd.DataFrame([
    {"Experiment": "Majority_Class_Benchmark", **dummy_metrics},
    {"Experiment": "Baseline_Unigram_TFIDF_LogReg", **metrics},
])
summary_df.to_csv(os.path.join(BASELINE_REPORTS_DIR, "evaluation_summary.csv"), index=False)

print("\n--- Step 6: Generating baseline charts ---")
save_charts(y_test, y_pred)

print("Baseline reports saved to:", BASELINE_REPORTS_DIR)
print("Baseline charts saved to:", BASELINE_CHARTS_DIR)
print("\nBaseline training finished successfully.")
