# Social Media Mood Analyzer

Social Media Mood Analyzer is a Natural Language Processing (NLP) system that
analyzes short social media-style texts and classifies them into four mood
categories: happy, sad, angry, and neutral.

## Pipeline

The project follows the proposal pipeline:

1. Load the GoEmotions dataset.
2. Map original emotion labels into happy, sad, angry, and neutral.
3. Preprocess text by lowercasing, removing URLs, mentions, hashtags,
   punctuation, and stopwords.
4. Resolve duplicate texts with conflicting labels.
5. Create one shared train/test split for all TF-IDF experiments.
6. Train and evaluate models using accuracy, precision, recall, F1-score, and
   macro F1-score.

## Run Order

Run from the project root:

```bash
source venv/bin/activate
python src/train_baseline.py
python src/train.py
```

Optional BERT fine-tuning:

```bash
python src/train_bert.py
```

New BERT runs save to `results/bert/model_fair` so they do not mix with
older checkpoints from previous dataset versions.

For a quick BERT smoke test, use a small sample:

```bash
BERT_MAX_SAMPLES=200 BERT_EPOCHS=1 python src/train_bert.py
```

## Current Fixed Results

The latest scripts use the same cleaned dataset and the same no-overlap
train/test split.

| Model | Accuracy | F1-score | Macro F1-score |
| --- | ---: | ---: | ---: |
| Majority-class benchmark | 38.12% | 21.05% | 13.80% |
| Baseline unigram TF-IDF + Logistic Regression | 70.30% | 69.29% | 63.03% |
| Final word+character TF-IDF + Logistic Regression | 70.57% | 70.05% | 64.60% |
