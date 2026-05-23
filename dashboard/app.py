"""
app.py
Streamlit dashboard for Social Media Mood Analyzer.
Run with: streamlit run dashboard/app.py
"""

import sys
import os
import re
import string

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import joblib
import pandas as pd

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

MODEL_PATH = "results/model.pkl"
VECTORIZER_PATH = "results/vectorizer.pkl"
DATA_PATH = "data/processed/mood_data.csv"
SUMMARY_PATH = "results/reports/evaluation_summary.csv"
REPORT_PATH = "results/reports/classification_report.txt"

# simple stopword list to avoid NLTK dependency in app
STOPWORDS = {
    "the", "is", "am", "are", "a", "an", "and", "to", "of", "in", "it", "this",
    "that", "for", "on", "with", "as", "was", "were", "be", "been", "being",
    "i", "im", "ive", "me", "my", "you", "your", "he", "she", "they", "them",
    "we", "our", "at", "by", "from", "or", "if", "but", "so", "do", "does",
    "did", "have", "has", "had", "not", "no", "just", "very", "can", "could",
    "would", "should", "will", "today"
}

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def preprocess_input(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and t.isalpha()]
    return " ".join(tokens)

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
        elif not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            st.error("Model or vectorizer not found. Please run training first.")
        else:
            model, vectorizer = load_model()
            cleaned = preprocess_input(user_input)

            if not cleaned.strip():
                st.warning("The text became empty after preprocessing. Try another input.")
            else:
                vec = vectorizer.transform([cleaned])
                mood = model.predict(vec)[0]

                st.markdown(
                    f'<div class="mood-box {mood}">{MOOD_EMOJI.get(mood, "")} {mood.upper()}</div>',
                    unsafe_allow_html=True
                )

                # show confidence only if supported
                if hasattr(model, "predict_proba"):
                    proba = dict(zip(model.classes_, model.predict_proba(vec)[0]))
                    st.markdown("#### Confidence per mood")
                    for m in sorted(proba, key=proba.get, reverse=True):
                        st.progress(float(proba[m]), text=f"{m.capitalize()}: {proba[m]*100:.1f}%")
                else:
                    st.info("This model does not provide probability scores.")

                st.markdown("#### Processed input")
                st.code(cleaned)

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
        "Confusion Matrix (Raw)": "results/charts/confusion_matrix_raw.png",
        "Confusion Matrix (Normalised)": "results/charts/confusion_matrix_normalised.png",
        "Per-Class Metrics": "results/charts/per_class_metrics.png",
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