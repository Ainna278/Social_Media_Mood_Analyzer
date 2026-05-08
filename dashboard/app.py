"""
app.py
Streamlit dashboard for Social Media Mood Analyzer.
Run with: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.evaluate import evaluate
from src.visualize import plot_all
from sklearn.model_selection import train_test_split

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Mood Analyzer",
    page_icon="💬",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .title { font-size: 2.5rem; font-weight: 800; color: #1a1a2e; }
    .subtitle { font-size: 1rem; color: #6c757d; margin-bottom: 2rem; }
    .mood-box {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .happy  { background-color: #fff3cd; color: #856404; border: 2px solid #ffc107; }
    .sad    { background-color: #cff4fc; color: #055160; border: 2px solid #0dcaf0; }
    .angry  { background-color: #f8d7da; color: #842029; border: 2px solid #dc3545; }
    .neutral{ background-color: #e2e3e5; color: #41464b; border: 2px solid #adb5bd; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    </style>
""", unsafe_allow_html=True)

MOOD_EMOJI = {"happy": "😊", "sad": "😢", "angry": "😡", "neutral": "😐"}
MOOD_LABELS = ["happy", "sad", "angry", "neutral"]

MODEL_PATH      = "results/model.pkl"
VECTORIZER_PATH = "results/vectorizer.pkl"
DATA_PATH       = "data/processed/mood_data.csv"


@st.cache_resource
def load_model():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def preprocess_input(text):
    """Basic cleaning for live input."""
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    nltk.download("punkt",     quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">💬 Social Media Mood Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">WID3002 Natural Language Processing · Group Project</p>', unsafe_allow_html=True)
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Mood", "📊 Evaluation Results", "📈 Visualizations"])


# ════════════════════════════════════════════════════════════════
# TAB 1 — Predict Mood
# ════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter a social media text")
    user_input = st.text_area(
        label="",
        placeholder="e.g. I can't believe how amazing today was!! 🎉",
        height=120
    )

    if st.button("Analyze Mood", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        elif not os.path.exists(MODEL_PATH):
            st.error("Model not found. Please run the training pipeline first.")
        else:
            model, vectorizer = load_model()
            cleaned = preprocess_input(user_input)
            vec     = vectorizer.transform([cleaned])
            mood    = model.predict(vec)[0]
            proba   = dict(zip(model.classes_, model.predict_proba(vec)[0]))

            emoji = MOOD_EMOJI.get(mood, "")
            st.markdown(
                f'<div class="mood-box {mood}">{emoji} {mood.upper()}</div>',
                unsafe_allow_html=True
            )

            st.markdown("#### Confidence per mood")
            for m in sorted(proba, key=proba.get, reverse=True):
                st.progress(proba[m], text=f"{m.capitalize()}: {proba[m]*100:.1f}%")

    # Example texts
    st.markdown("---")
    st.markdown("**Try an example:**")
    examples = {
        "😊 Happy"  : "I just got promoted today!! Best day of my life!!",
        "😢 Sad"    : "I miss my old friends so much, everything feels empty",
        "😡 Angry"  : "This is absolutely ridiculous, I can't believe this happened",
        "😐 Neutral": "The meeting has been rescheduled to tomorrow at 3pm",
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
    report_path  = "results/reports/classification_report.txt"
    summary_path = "results/reports/evaluation_summary.csv"

    if not os.path.exists(summary_path):
        st.info("No evaluation results yet. Run the training pipeline first.")
    else:
        summary = pd.read_csv(summary_path)

        st.subheader("Overall Metrics")
        col1, col2, col3, col4 = st.columns(4)
        metrics = {
            "Accuracy":  ("🎯", summary["Accuracy"].values[0]),
            "Precision": ("🔎", summary["Precision"].values[0]),
            "Recall":    ("📡", summary["Recall"].values[0]),
            "F1-Score":  ("⚖️", summary["F1-Score"].values[0]),
        }
        for col, (label, (icon, val)) in zip([col1, col2, col3, col4], metrics.items()):
            col.metric(label=f"{icon} {label}", value=f"{val*100:.2f}%")

        if os.path.exists(report_path):
            st.subheader("Full Classification Report")
            with open(report_path) as f:
                st.code(f.read(), language=None)


# ════════════════════════════════════════════════════════════════
# TAB 3 — Visualizations
# ════════════════════════════════════════════════════════════════
with tab3:
    charts = {
        "Confusion Matrix (Raw)"       : "results/charts/confusion_matrix_raw.png",
        "Confusion Matrix (Normalised)": "results/charts/confusion_matrix_normalised.png",
        "Per-Class Metrics"            : "results/charts/per_class_metrics.png",
    }

    available = {k: v for k, v in charts.items() if os.path.exists(v)}

    if not available:
        st.info("No charts found. Run the training pipeline first.")
    else:
        for title, path in available.items():
            st.subheader(title)
            st.image(path, use_column_width=True)
            st.divider()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<center><small>WID3002 NLP · Social Media Mood Analyzer · Group Project</small></center>",
    unsafe_allow_html=True
)