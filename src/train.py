# src/train.py
import os
import string
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# 1. Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 2. Setup paths
DATASET_DIR = r"C:\Users\AINNA HUMAIRAK\.cache\kagglehub\datasets\debarshichanda\goemotions\versions\6\data\full_dataset"
PROCESSED_DATA_PATH = "data/processed/mood_data.csv"

# Ensure output directories exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("results/charts", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

print("--- Step 1: Loading and Merging Datasets ---")
all_files = glob.glob(os.path.join(DATASET_DIR, "goemotions_*.csv"))
df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
df = df[df['example_very_unclear'] == False].copy()

# 3. Label Mapping
emotion_mapping = {
    'joy': 'happy', 'amusement': 'happy', 'excitement': 'happy', 'love': 'happy', 
    'optimism': 'happy', 'gratitude': 'happy', 'pride': 'happy', 'relief': 'happy', 
    'admiration': 'happy', 'caring': 'happy', 'approval': 'happy',
    'sadness': 'sad', 'grief': 'sad', 'disappointment': 'sad', 'remorse': 'sad', 'embarrassment': 'sad',
    'anger': 'angry', 'annoyance': 'angry', 'disapproval': 'angry', 'disgust': 'angry',
    'neutral': 'neutral', 'confusion': 'neutral', 'curiosity': 'neutral', 
    'realization': 'neutral', 'surprise': 'neutral', 'fear': 'neutral', 'nervousness': 'neutral', 'desire': 'neutral'
}

def get_target_mood(row):
    for emotion, mood in emotion_mapping.items():
        if row[emotion] == 1:
            return mood
    return 'neutral'

df['mood'] = df.apply(get_target_mood, axis=1)
df_clean = df[['text', 'mood']].copy()

print("--- Step 2: Preprocessing Text ---")
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w not in stop_words])

df_clean['processed_text'] = df_clean['text'].apply(preprocess_text)

# Save processed text data for dashboard fallback load
df_clean.to_csv(PROCESSED_DATA_PATH, index=False)

print("--- Step 3: Feature Extraction & Training ---")
X = df_clean['processed_text']
y = df_clean['mood']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save artifacts
joblib.dump(model, 'results/model.pkl')
joblib.dump(vectorizer, 'results/vectorizer.pkl')

print("--- Step 4: Generating Reports & Metrics ---")
y_pred = model.predict(X_test_tfidf)

# Generate Text Classification Report
report = classification_report(y_test, y_pred)
with open("results/reports/classification_report.txt", "w") as f:
    f.write(report)

# Generate Evaluation Summary CSV
acc = accuracy_score(y_test, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
summary_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1-Score": f1
}])
summary_df.to_csv("results/reports/evaluation_summary.csv", index=False)

print("--- Step 5: Generating Dashboard Charts ---")
labels = ['happy', 'sad', 'angry', 'neutral']
cm = confusion_matrix(y_test, y_pred, labels=labels)

# 1. Raw Confusion Matrix Chart
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Raw)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/charts/confusion_matrix_raw.png')
plt.close()

# 2. Normalized Confusion Matrix Chart
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(6, 5))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix (Normalised)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('results/charts/confusion_matrix_normalised.png')
plt.close()

# 3. Per-Class Metrics Bar Chart
p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, labels=labels)
metrics_df = pd.DataFrame({'Precision': p, 'Recall': r, 'F1-Score': f}, index=labels)
metrics_df.plot(kind='bar', figsize=(8, 5))
plt.title('Per-Class Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('results/charts/per_class_metrics.png')
plt.close()

print("Training script finished successfully! Everything is ready for the dashboard.")