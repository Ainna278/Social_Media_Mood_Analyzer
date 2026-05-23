import os
import re
import glob
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# =========================
# Paths
# =========================
DATASET_DIR = "archive/data/full_dataset"
PROCESSED_DATA_PATH = "data/processed/mood_data.csv"

os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

# =========================
# Load raw GoEmotions files
# =========================
print("--- Step 1: Loading Raw GoEmotions Dataset ---")
all_files = glob.glob(os.path.join(DATASET_DIR, "goemotions_*.csv"))
print("Files found:", all_files)

if not all_files:
    raise FileNotFoundError(
        f"No GoEmotions CSV files found in: {DATASET_DIR}\n"
        f"Make sure goemotions_1.csv, goemotions_2.csv, etc. are inside that folder."
    )

df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

if "example_very_unclear" in df.columns:
    df = df[df["example_very_unclear"] == False].copy()

print("Raw dataset loaded successfully.")
print("Columns:", df.columns.tolist())

# =========================
# Final label mapping
# =========================
emotion_mapping = {
    "joy": "happy",
    "amusement": "happy",
    "excitement": "happy",
    "love": "happy",
    "gratitude": "happy",
    "optimism": "happy",
    "admiration": "happy",

    "sadness": "sad",
    "grief": "sad",
    "disappointment": "sad",
    "remorse": "sad",
    "embarrassment": "sad",

    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "angry",

    "neutral": "neutral"
}

target_emotions = list(emotion_mapping.keys())

def map_single_mood(row):
    active_moods = [emotion_mapping[e] for e in target_emotions if row.get(e, 0) == 1]
    active_moods = list(set(active_moods))

    if len(active_moods) == 1:
        return active_moods[0]
    return np.nan

df["mood"] = df.apply(map_single_mood, axis=1)
df_clean = df[["text", "mood"]].dropna().copy()

print("\nLabel distribution after clean mapping:")
print(df_clean["mood"].value_counts())

# =========================
# Text preprocessing
# =========================
print("\n--- Step 2: Preprocessing Text ---")
print("Using final light preprocessing only. No NLTK download/tokenization needed.")

def light_preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_clean["light_text"] = df_clean["text"].apply(light_preprocess_text)
df_clean = df_clean[df_clean["light_text"].str.strip() != ""]

df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")

# =========================
# Train-test split
# =========================
print("\n--- Step 3: Splitting Data ---")
X = df_clean["light_text"]
y = df_clean["mood"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# Final TF-IDF + Logistic Regression Model
# =========================
print("\n--- Step 4 & 5: Training Final TF-IDF + Logistic Regression Model ---")
print("Final selected setup: light text + word TF-IDF + character TF-IDF + Logistic Regression")

config = {
    "experiment": "Final_I13_light_more_word_C15",
"word_max_features": 60000,
"word_ngram_range": (1, 2),
"char_max_features": 70000,
"char_ngram_range": (3, 5),
"min_df": 1,
"char_min_df": 2,
"max_df": 0.95,
"C": 1.5,
"class_weight": None
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
            sublinear_tf=True
        )
    ),
    (
        "char_tfidf",
        TfidfVectorizer(
            analyzer="char_wb",
            max_features=config["char_max_features"],
            ngram_range=config["char_ngram_range"],
            min_df=config["char_min_df"],
            max_df=config["max_df"],
            sublinear_tf=True
        )
    )
])

print("Fitting TF-IDF vectorizer... this may take a few minutes.")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print("TF-IDF vectorizer finished.")

model = LogisticRegression(
    max_iter=3000,
    random_state=42,
    C=config["C"],
    class_weight=config["class_weight"],
    solver="lbfgs"
)

print("Training Logistic Regression model... this is the slowest step.")
model.fit(X_train_tfidf, y_train)

print("Model training finished. Predicting test set...")
y_pred = model.predict(X_test_tfidf)

joblib.dump(model, "results/model.pkl")
joblib.dump(vectorizer, "results/vectorizer.pkl")

print("Model saved to: results/model.pkl")
print("Vectorizer saved to: results/vectorizer.pkl")

# =========================
# Evaluation
# =========================
print("\n--- Step 6: Evaluating Final Model ---")

acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)
macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

report = classification_report(y_test, y_pred, zero_division=0)

with open("results/reports/classification_report.txt", "w") as f:
    f.write(report)

summary_df = pd.DataFrame([{
    "Best Experiment": config["experiment"],
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1,
    "Macro Precision": macro_prec,
    "Macro Recall": macro_rec,
    "Macro F1-Score": macro_f1
}])

summary_df.to_csv("results/reports/evaluation_summary.csv", index=False)
summary_df.to_csv("results/reports/experiment_log.csv", index=False)

print("Classification report saved to: results/reports/classification_report.txt")
print("Evaluation summary saved to: results/reports/evaluation_summary.csv")

# =========================
# Charts
# =========================
print("\n--- Step 7: Generating Charts ---")
labels = ["happy", "sad", "angry", "neutral"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Confusion Matrix (Raw)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("results/charts/confusion_matrix_raw.png")
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
    xticklabels=labels,
    yticklabels=labels
)
plt.title("Confusion Matrix (Normalised)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("results/charts/confusion_matrix_normalised.png")
plt.close()

p, r, f, _ = precision_recall_fscore_support(
    y_test, y_pred, labels=labels, zero_division=0
)

metrics_df = pd.DataFrame(
    {
        "Precision": p,
        "Recall": r,
        "F1-Score": f
    },
    index=labels
)

metrics_df.plot(kind="bar", figsize=(8, 5))
plt.title("Per-Class Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("results/charts/per_class_metrics.png")
plt.close()

print("Charts saved in: results/charts/")
print("\nTraining script finished successfully.")