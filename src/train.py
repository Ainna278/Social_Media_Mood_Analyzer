import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion

from pipeline_utils import MOOD_LABELS, get_train_test_split, print_dataset_summary


REPORTS_DIR = "results/reports"
CHARTS_DIR = "results/charts"
MODEL_PATH = "results/model.pkl"
VECTORIZER_PATH = "results/vectorizer.pkl"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


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
    plt.title("Confusion Matrix (Raw)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix_raw.png"))
    plt.close()

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype("float") / row_sums

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS)
    plt.title("Confusion Matrix (Normalised)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix_normalised.png"))
    plt.close()

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=MOOD_LABELS, zero_division=0)
    metrics_df = pd.DataFrame({"Precision": p, "Recall": r, "F1-Score": f1}, index=MOOD_LABELS)

    metrics_df.plot(kind="bar", figsize=(8, 5))
    plt.title("Per-Class Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "per_class_metrics.png"))
    plt.close()


print("--- Step 1: Loading shared processed dataset and split ---")
train_df, test_df, stats = get_train_test_split()
print_dataset_summary(stats, train_df, test_df)

X_train = train_df["clean_text"]
y_train = train_df["mood"]
X_test = test_df["clean_text"]
y_test = test_df["mood"]

print("\n--- Step 2: Final TF-IDF feature extraction ---")
config = {
    "experiment": "Final_WordChar_TFIDF_LogReg_C15",
    "word_max_features": 60000,
    "word_ngram_range": (1, 2),
    "char_max_features": 70000,
    "char_ngram_range": (3, 5),
    "min_df": 1,
    "char_min_df": 2,
    "max_df": 0.95,
    "C": 1.5,
    "class_weight": None,
}

vectorizer = FeatureUnion([
    (
        "word_tfidf",
        TfidfVectorizer(
            analyzer="word",
            max_features=config["word_max_features"],
            ngram_range=config["word_ngram_range"],
            min_df=config["min_df"],
            max_df=config["max_df"],
            sublinear_tf=True,
        ),
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            max_features=config["char_max_features"],
            ngram_range=config["char_ngram_range"],
            min_df=config["char_min_df"],
            max_df=config["max_df"],
            sublinear_tf=True,
        ),
    ),
])

print("Fitting TF-IDF vectorizer...")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF vectorizer finished.")

print("\n--- Step 3: Training final Logistic Regression model ---")
model = LogisticRegression(
    max_iter=3000,
    random_state=42,
    C=config["C"],
    class_weight=config["class_weight"],
    solver="lbfgs",
)
model.fit(X_train_tfidf, y_train)

print("Model training finished. Predicting test set...")
y_pred = model.predict(X_test_tfidf)

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print("Model saved to:", MODEL_PATH)
print("Vectorizer saved to:", VECTORIZER_PATH)

print("\n--- Step 4: Evaluating final model ---")
metrics = evaluate_predictions(y_test, y_pred)
for key in ["Accuracy", "Precision", "Recall", "F1-Score", "Macro F1-Score"]:
    print(f"{key}: {metrics[key]:.4f}")

report = classification_report(y_test, y_pred, labels=MOOD_LABELS, zero_division=0)

with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

summary_df = pd.DataFrame([{ "Best Experiment": config["experiment"], **metrics }])
summary_df.to_csv(os.path.join(REPORTS_DIR, "evaluation_summary.csv"), index=False)
summary_df.to_csv(os.path.join(REPORTS_DIR, "experiment_log.csv"), index=False)

baseline_summary_path = "results/baseline/reports/evaluation_summary.csv"
if os.path.exists(baseline_summary_path):
    baseline_summary = pd.read_csv(baseline_summary_path)
    final_summary = pd.DataFrame([{ "Experiment": config["experiment"], **metrics }])
    comparison = pd.concat([baseline_summary, final_summary], ignore_index=True, sort=False)
    comparison.to_csv(os.path.join(REPORTS_DIR, "experiment_comparison.csv"), index=False)

print("\n--- Step 5: Generating charts ---")
save_charts(y_test, y_pred)

print("Reports saved to:", REPORTS_DIR)
print("Charts saved to:", CHARTS_DIR)
print("\nTraining script finished successfully.")
