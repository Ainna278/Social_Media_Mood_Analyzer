import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# =========================
# Paths
# =========================
DATA_PATH = "data/processed/mood_data.csv"
RESULTS_DIR = "results/bert"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MODEL_DIR = os.path.join(RESULTS_DIR, "model")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# Load dataset
# =========================
print("--- Step 1: Loading processed dataset ---")
df = pd.read_csv(DATA_PATH)

# choose input column
if "text" in df.columns:
    text_col = "text"
elif "light_text" in df.columns:
    text_col = "light_text"
elif "processed_text" in df.columns:
    text_col = "processed_text"
else:
    raise ValueError("No valid text column found. Expected one of: text, light_text, processed_text")

if "mood" not in df.columns:
    raise ValueError("Missing required column: mood")

df = df[[text_col, "mood"]].dropna().copy()
df[text_col] = df[text_col].astype(str).str.strip()
df = df[df[text_col] != ""]

print("Dataset loaded successfully.")
print("Using text column:", text_col)
print(df["mood"].value_counts())

# =========================
# Label encoding
# =========================
label2id = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "neutral": 3
}
id2label = {v: k for k, v in label2id.items()}

df = df[df["mood"].isin(label2id.keys())].copy()
df["label"] = df["mood"].map(label2id)

# =========================
# Split data
# =========================
print("\n--- Step 2: Train-test split ---")
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# optional validation split from train
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["label"]
)

print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# =========================
# Hugging Face Dataset
# =========================
train_ds = Dataset.from_pandas(train_df[[text_col, "label"]], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[[text_col, "label"]], preserve_index=False)
test_ds = Dataset.from_pandas(test_df[[text_col, "label"]], preserve_index=False)

# =========================
# Tokenizer and model
# =========================
print("\n--- Step 3: Loading BERT tokenizer and model ---")
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(batch):
    return tokenizer(
        batch[text_col],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

columns_to_keep = ["input_ids", "attention_mask", "label"]
train_ds.set_format(type="torch", columns=columns_to_keep)
val_ds.set_format(type="torch", columns=columns_to_keep)
test_ds.set_format(type="torch", columns=columns_to_keep)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

# =========================
# Metrics
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1
    }

# =========================
# Training config
# =========================
print("\n--- Step 4: Training BERT ---")
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# =========================
# Final evaluation on test set
# =========================
print("\n--- Step 5: Evaluating on test set ---")
pred_output = trainer.predict(test_ds)
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)

acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Macro F1 : {macro_f1:.4f}")

# =========================
# Save model/tokenizer
# =========================
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("BERT model saved to:", MODEL_DIR)

# =========================
# Save reports
# =========================
report = classification_report(
    y_true,
    y_pred,
    target_names=[id2label[i] for i in range(4)],
    zero_division=0
)

with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

summary_df = pd.DataFrame([{
    "Model": "BERT Fine-tuned",
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1,
    "Macro Precision": macro_prec,
    "Macro Recall": macro_rec,
    "Macro F1-Score": macro_f1
}])

summary_df.to_csv(os.path.join(REPORTS_DIR, "evaluation_summary.csv"), index=False)

# =========================
# Charts
# =========================
print("\n--- Step 6: Generating charts ---")
labels = ["happy", "sad", "angry", "neutral"]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.title("BERT Confusion Matrix (Raw)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix_raw.png"))
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
plt.title("BERT Confusion Matrix (Normalised)")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix_normalised.png"))
plt.close()

p, r, f, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
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
plt.title("BERT Per-Class Metrics")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "per_class_metrics.png"))
plt.close()

print("BERT reports saved to:", REPORTS_DIR)
print("BERT charts saved to:", CHARTS_DIR)
print("\nBERT fine-tuning finished successfully.")