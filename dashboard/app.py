"""
app.py
Streamlit dashboard for Social Media Mood Analyzer.
Run with: streamlit run dashboard/app.py
"""

import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from pipeline_utils import light_preprocess_text

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Mood Analyzer",
    page_icon="💬",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Title & subtitle */
.title {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}

.subtitle {
    font-size: 1rem;
    opacity: 0.8;
    margin-bottom: 1.5rem;
}

/* Text area */
.stTextArea textarea {
    border-radius: 12px !important;
    border: 1px solid rgba(128, 128, 128, 0.35) !important;
    background-color: rgba(127, 127, 127, 0.08) !important;
    color: inherit !important;
}

/* Main primary button */
div.stButton > button[kind="primary"] {
    background-color: #ef4444 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.2rem !important;
    font-weight: 600 !important;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #dc2626 !important;
    color: white !important;
}

/* Default buttons (example buttons) */
div.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(128, 128, 128, 0.35) !important;
    background-color: transparent !important;
    color: inherit !important;
    font-weight: 500 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 12px;
    padding: 1rem;
    background-color: rgba(127, 127, 127, 0.06);
}

/* Code/report box */
.stCodeBlock, pre {
    border-radius: 12px !important;
}

/* Mood result box */
.mood-box {
    padding: 1.5rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    margin-top: 1rem;
    border: 1px solid rgba(128, 128, 128, 0.3);
}

/* Mood colors that still work in both themes */
.happy  { background-color: rgba(234, 179, 8, 0.18); }
.sad    { background-color: rgba(56, 189, 248, 0.18); }
.angry  { background-color: rgba(239, 68, 68, 0.18); }
.neutral{ background-color: rgba(107, 114, 128, 0.18); }
</style>
""", unsafe_allow_html=True)

MOOD_EMOJI = {"happy": "😊", "sad": "😢", "angry": "😡", "neutral": "😐"}
MOOD_LABELS = ["happy", "sad", "angry", "neutral"]
label2id = {label: idx for idx, label in enumerate(MOOD_LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

BERT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = "data/processed/mood_data.csv"
SUMMARY_PATH = "results/bert/reports/evaluation_summary.csv"
REPORT_PATH = "results/bert/reports/classification_report.txt"

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    return model, tokenizer

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def preprocess_input(text: str) -> str:
    return light_preprocess_text(text)

def apply_postprocessing_rules(raw_text: str, predicted_mood: str):
    text = str(raw_text).lower()
    if predicted_mood == "happy" and re.search(r"\b(not|never|no)\s+(so\s+|very\s+|really\s+)?happy\b", text):
        return "sad", "Adjusted because the text contains explicit negation such as 'not happy'."
    if predicted_mood == "happy" and re.search(r"\b(not|never|no)\s+(feeling\s+)?good\b", text):
        return "sad", "Adjusted because the text contains explicit negation such as 'not good'."
    if predicted_mood == "happy" and re.search(
        r"\b(but|though|although|however)\b.*\b(empty|tired|sad|lonely|depressed|terrible|awful|hurt|crying|miserable)\b",
        text,
    ):
        return "sad", "Adjusted because the text has a contrast phrase where the negative feeling appears after 'but'."
    if predicted_mood in {"neutral", "happy"} and re.search(
        r"\b(waited|waiting|wasted)\b.*\b(cancelled|canceled|cancel|last minute|late|ignored)\b",
        text,
    ):
        return "angry", "Adjusted because the text describes a common complaint pattern: waiting followed by cancellation or delay."
    return predicted_mood, None

def adjust_probabilities(proba: dict, raw_mood: str, final_mood: str) -> dict:
    if raw_mood == final_mood:
        return proba

    adjusted = proba.copy()
    raw_score = adjusted.get(raw_mood, 0.0)
    final_score = adjusted.get(final_mood, 0.0)
    adjusted[final_mood] = max(raw_score, final_score)
    adjusted[raw_mood] = min(raw_score, final_score)
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {mood: score / total for mood, score in adjusted.items()}
    return adjusted

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">💬 Social Media Mood Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">WID3002 Natural Language Processing · Group Project</p>', unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Mood", "📊 Evaluation Results", "📈 Visualizations"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — Predict Mood
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter a social media text")
    user_input = st.text_area(
        label="",
        placeholder="e.g. I can't believe how amazing today was!!",
        height=120
    )

    if st.button("Analyze Mood", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        elif not os.path.exists(BERT_MODEL_PATH):
            st.error(f"BERT model not found at {BERT_MODEL_PATH}. Please train BERT first.")
        else:
            model, tokenizer = load_model()
            preprocessed = preprocess_input(user_input)

            if not preprocessed.strip():
                st.warning("The text became empty after preprocessing. Try another input.")
            else:
                inputs = tokenizer(
                    preprocessed,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=128
                )

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits[0].cpu().numpy()

                proba_scores = softmax(logits)
                raw_mood_idx = np.argmax(logits)
                raw_mood = id2label[raw_mood_idx]

                mood, rule_note = apply_postprocessing_rules(user_input, raw_mood)

                st.markdown(
                    f'<div class="mood-box {mood}">{MOOD_EMOJI.get(mood, "")} {mood.upper()}</div>',
                    unsafe_allow_html=True
                )

                proba_dict = {id2label[i]: float(proba_scores[i]) for i in range(len(MOOD_LABELS))}
                proba_dict = adjust_probabilities(proba_dict, raw_mood, mood)
                st.markdown("#### Confidence per mood")
                for m in sorted(proba_dict, key=proba_dict.get, reverse=True):
                    st.progress(proba_dict[m], text=f"{m.capitalize()}: {proba_dict[m]*100:.1f}%")

                st.markdown("#### Processed input")
                st.code(preprocessed)

    st.markdown("---")
    st.markdown("**Try an example:**")
    examples = {
        "😊 Happy": "I just got promoted today, best day ever!",
        "😢 Sad": "I feel tired and empty today",
        "😡 Angry": "Today is so annoying and frustrating",
        "😐 Neutral": "The meeting has been moved to tomorrow at 3pm"
    }

    cols = st.columns(4)
    for i, (label, text) in enumerate(examples.items()):
        if cols[i].button(label, use_container_width=True):
            st.session_state["example_text"] = text
            st.rerun()

    if "example_text" in st.session_state:
        st.info(f'**Example:** {st.session_state["example_text"]}')

# ════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Results
# ════════════════════════════════════════════════════════════════
with tab2:
    if not os.path.exists(SUMMARY_PATH):
        st.info("No evaluation results yet. Run the training pipeline first.")
    else:
        summary = pd.read_csv(SUMMARY_PATH)

        st.subheader("Overall Metrics")
        col1, col2, col3, col4 = st.columns(4)
        metrics = {
            "Accuracy": summary["Accuracy"].values[0],
            "Precision": summary["Precision"].values[0],
            "Recall": summary["Recall"].values[0],
            "F1-Score": summary["F1-Score"].values[0],
        }

        for col, (label, val) in zip([col1, col2, col3, col4], metrics.items()):
            col.metric(label=label, value=f"{val*100:.2f}%")

        if os.path.exists(REPORT_PATH):
            st.subheader("Full Classification Report")
            with open(REPORT_PATH, "r") as f:
                st.code(f.read())

# ════════════════════════════════════════════════════════════════
# TAB 3 — Visualizations
# ════════════════════════════════════════════════════════════════
with tab3:
    charts = {
        "Confusion Matrix (Raw)": "results/bert/charts/confusion_matrix_raw.png",
        "Confusion Matrix (Normalised)": "results/bert/charts/confusion_matrix_normalised.png",
        "Per-Class Metrics": "results/bert/charts/per_class_metrics.png",
    }

    available = {k: v for k, v in charts.items() if os.path.exists(v)}

    if not available:
        st.info("No charts found. Run the training pipeline first.")
    else:
        for title, path in available.items():
            st.subheader(title)
            st.image(path, use_container_width=True)
            st.divider()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<center><small>WID3002 NLP · Social Media Mood Analyzer · Group Project</small></center>",
    unsafe_allow_html=True
)
