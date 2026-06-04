import os
import re
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_DIR = "archive/data/full_dataset"
PROCESSED_DATA_PATH = "data/processed/mood_data.csv"
SPLIT_DIR = "data/processed/splits"
RANDOM_STATE = 42
MOOD_LABELS = ["happy", "sad", "angry", "neutral"]

EMOTION_MAPPING = {
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
    "neutral": "neutral",
}

STOPWORDS = {
    "the", "is", "am", "are", "a", "an", "and", "to", "of", "in", "it",
    "this", "that", "for", "on", "with", "as", "was", "were", "be",
    "been", "being", "i", "im", "ive", "me", "my", "you", "your", "he",
    "she", "they", "them", "we", "our", "at", "by", "from", "or", "if",
    "but", "so", "do", "does", "did", "have", "has", "had", "just",
    "very", "can", "could", "would", "should", "will", "today",
}


def preprocess_text(text):
    """Proposal-aligned preprocessing for TF-IDF models and dashboard input."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in text.split() if token not in STOPWORDS and token.isalpha()]
    return " ".join(tokens)


def light_preprocess_text(text):
    """Light preprocessing for BERT, which should keep natural sentence structure."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _map_single_mood(row):
    active_moods = [EMOTION_MAPPING[e] for e in EMOTION_MAPPING if row.get(e, 0) == 1]
    active_moods = list(set(active_moods))
    if len(active_moods) == 1:
        return active_moods[0]
    return np.nan


def _resolve_duplicate_labels(df):
    rows = []
    dropped_ties = 0

    for clean_text, group in df.groupby("clean_text", sort=False):
        counts = group["mood"].value_counts()
        if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
            dropped_ties += len(group)
            continue

        chosen_mood = counts.index[0]
        first = group.iloc[0].copy()
        first["mood"] = chosen_mood
        rows.append(first)

    resolved = pd.DataFrame(rows).reset_index(drop=True)
    return resolved, dropped_ties


def build_processed_dataset(dataset_dir=DATASET_DIR, output_path=PROCESSED_DATA_PATH):
    import glob

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(dataset_dir, "goemotions_*.csv")))
    if not all_files:
        raise FileNotFoundError(
            f"No GoEmotions CSV files found in: {dataset_dir}. "
            "Make sure goemotions_1.csv, goemotions_2.csv, etc. are inside that folder."
        )

    df = pd.concat([pd.read_csv(path) for path in all_files], ignore_index=True)
    raw_rows = len(df)

    if "example_very_unclear" in df.columns:
        df = df[df["example_very_unclear"] == False].copy()

    df["mood"] = df.apply(_map_single_mood, axis=1)
    df = df[["text", "mood"]].dropna().copy()
    mapped_rows = len(df)

    df["clean_text"] = df["text"].apply(preprocess_text)
    df["bert_text"] = df["text"].apply(light_preprocess_text)
    df = df[(df["clean_text"].str.strip() != "") & (df["bert_text"].str.strip() != "")]

    before_dedup = len(df)
    df, dropped_ties = _resolve_duplicate_labels(df)
    df = df[["text", "clean_text", "bert_text", "mood"]].reset_index(drop=True)
    df.to_csv(output_path, index=False)

    stats = {
        "raw_rows": raw_rows,
        "mapped_rows": mapped_rows,
        "rows_before_dedup": before_dedup,
        "rows_after_dedup": len(df),
        "dropped_tie_rows": dropped_ties,
        "label_counts": df["mood"].value_counts().to_dict(),
        "source_files": all_files,
    }
    return df, stats


def load_or_build_processed_dataset(force_rebuild=False):
    if force_rebuild or not os.path.exists(PROCESSED_DATA_PATH):
        return build_processed_dataset()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    expected_cols = {"text", "clean_text", "bert_text", "mood"}
    if not expected_cols.issubset(df.columns):
        return build_processed_dataset()

    stats = {
        "rows_after_dedup": len(df),
        "label_counts": df["mood"].value_counts().to_dict(),
        "source_files": [],
    }
    return df, stats


def get_train_test_split(force_rebuild=False):
    df, stats = load_or_build_processed_dataset(force_rebuild=force_rebuild)
    os.makedirs(SPLIT_DIR, exist_ok=True)

    train_path = os.path.join(SPLIT_DIR, "train.csv")
    test_path = os.path.join(SPLIT_DIR, "test.csv")

    if force_rebuild or not (os.path.exists(train_path) and os.path.exists(test_path)):
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=df["mood"],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    else:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

    return train_df, test_df, stats


def print_dataset_summary(stats, train_df=None, test_df=None):
    print("Dataset summary:")
    for key in ["raw_rows", "mapped_rows", "rows_before_dedup", "rows_after_dedup", "dropped_tie_rows"]:
        if key in stats:
            print(f"- {key}: {stats[key]}")
    print("- label_counts:")
    for label, count in stats.get("label_counts", {}).items():
        print(f"  {label}: {count}")
    if train_df is not None and test_df is not None:
        overlap = set(train_df["clean_text"]).intersection(set(test_df["clean_text"]))
        print(f"- train size: {len(train_df)}")
        print(f"- test size: {len(test_df)}")
        print(f"- train/test duplicate clean_text overlap: {len(overlap)}")
