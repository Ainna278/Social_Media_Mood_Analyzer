import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from pipeline_utils import MOOD_LABELS, get_train_test_split, print_dataset_summary


RESULTS_DIR = "results/bert"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")
MODEL_DIR = os.getenv("BERT_OUTPUT_DIR", os.path.join(RESULTS_DIR, "model_fair"))

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

label2id = {label: idx for idx, label in enumerate(MOOD_LABELS)}
id2label = {idx: label for label, idx in label2id.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def save_charts(y_true, y_pred):
    label_ids = list(range(len(MOOD_LABELS)))
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS)
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
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=MOOD_LABELS, yticklabels=MOOD_LABELS)
    plt.title("BERT Confusion Matrix (Normalised)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "confusion_matrix_normalised.png"))
    plt.close()

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=label_ids, zero_division=0)
    metrics_df = pd.DataFrame({"Precision": p, "Recall": r, "F1-Score": f1}, index=MOOD_LABELS)

    metrics_df.plot(kind="bar", figsize=(8, 5))
    plt.title("BERT Per-Class Metrics")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, "per_class_metrics.png"))
    plt.close()


print("--- Step 1: Loading shared processed dataset and split ---")
train_df, test_df, stats = get_train_test_split()
print_dataset_summary(stats, train_df, test_df)

max_samples = int(os.getenv("BERT_MAX_SAMPLES", "0"))
if max_samples > 0:
    train_df = train_df.sample(min(max_samples, len(train_df)), random_state=42)
    test_df = test_df.sample(min(max_samples // 4, len(test_df)), random_state=42)
    print(f"Debug mode: using {len(train_df)} training rows and {len(test_df)} test rows.")

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["mood"],
)

for frame in [train_df, val_df, test_df]:
    frame["label"] = frame["mood"].map(label2id)

print("\n--- Step 2: Creating Hugging Face datasets ---")
train_ds = Dataset.from_pandas(train_df[["bert_text", "label"]], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[["bert_text", "label"]], preserve_index=False)
test_ds = Dataset.from_pandas(test_df[["bert_text", "label"]], preserve_index=False)

print("\n--- Step 3: Loading BERT tokenizer and model ---")
model_name = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(batch):
    return tokenizer(batch["bert_text"], truncation=True, padding="max_length", max_length=128)


train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

columns_to_keep = ["input_ids", "attention_mask", "label"]
train_ds.set_format(type="torch", columns=columns_to_keep)
val_ds.set_format(type="torch", columns=columns_to_keep)
test_ds.set_format(type="torch", columns=columns_to_keep)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(MOOD_LABELS),
    id2label=id2label,
    label2id=label2id,
)

device_note = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Training device detected:", device_note)

print("\n--- Step 4: Training BERT ---")
training_args = TrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=float(os.getenv("BERT_LR", "2e-5")),
    per_device_train_batch_size=int(os.getenv("BERT_TRAIN_BATCH_SIZE", "8")),
    per_device_eval_batch_size=int(os.getenv("BERT_EVAL_BATCH_SIZE", "8")),
    num_train_epochs=float(os.getenv("BERT_EPOCHS", "3")),
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

resume_checkpoint = None
if os.getenv("BERT_RESUME", "0") == "1":
    resume_checkpoint = get_last_checkpoint(MODEL_DIR)
    print("Resume checkpoint:", resume_checkpoint)

trainer.train(resume_from_checkpoint=resume_checkpoint)

print("\n--- Step 5: Evaluating on test set ---")
pred_output = trainer.predict(test_ds)
y_true = pred_output.label_ids
y_pred = np.argmax(pred_output.predictions, axis=1)

metrics = compute_metrics((pred_output.predictions, y_true))

print(f"Accuracy : {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall   : {metrics['recall']:.4f}")
print(f"F1-score : {metrics['f1']:.4f}")
print(f"Macro F1 : {metrics['macro_f1']:.4f}")

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print("BERT model saved to:", MODEL_DIR)

report = classification_report(
    y_true,
    y_pred,
    target_names=MOOD_LABELS,
    labels=list(range(len(MOOD_LABELS))),
    zero_division=0,
)

with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

summary_df = pd.DataFrame([{
    "Model": "BERT Fine-tuned",
    "Accuracy": metrics["accuracy"],
    "Precision": metrics["precision"],
    "Recall": metrics["recall"],
    "F1-Score": metrics["f1"],
    "Macro Precision": metrics["macro_precision"],
    "Macro Recall": metrics["macro_recall"],
    "Macro F1-Score": metrics["macro_f1"],
}])
summary_df.to_csv(os.path.join(REPORTS_DIR, "evaluation_summary.csv"), index=False)

print("\n--- Step 6: Generating charts ---")
save_charts(y_true, y_pred)

print("BERT reports saved to:", REPORTS_DIR)
print("BERT charts saved to:", CHARTS_DIR)
print("\nBERT fine-tuning finished successfully.")
