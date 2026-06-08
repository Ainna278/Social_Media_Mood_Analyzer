# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs for torch/scipy/pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Layer 1: Heavy ML deps (torch + transformers) ─────────────────────────────
# Cached until requirements-heavy.txt changes — changing app code never re-runs this.
# BuildKit cache mount keeps downloaded wheels on the build host so re-runs are fast.
COPY requirements-heavy.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-heavy.txt \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

# ── Layer 2: Lighter app deps ─────────────────────────────────────────────────
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# ── Layer 3: Pre-download model from HuggingFace Hub ─────────────────────────
# Baked into the image so teammates get instant startup — no Hub download at runtime.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python -c "\
from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
AutoTokenizer.from_pretrained('touhidulai/Bert-base-Uncased'); \
AutoModelForSequenceClassification.from_pretrained('touhidulai/Bert-base-Uncased')"

# ── Layer 4: Application code (changes most often — kept last) ────────────────
COPY src/          ./src/
COPY dashboard/    ./dashboard/
COPY data/         ./data/
COPY results/      ./results/

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.headless=true"]
