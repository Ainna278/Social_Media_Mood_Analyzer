# src/model.py

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def build_model():
    """
    Initialize Logistic Regression model.

    Returns:
        LogisticRegression instance
    """
    return LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )


def train_model(model, X_train, y_train):
    """
    Train model on training data.

    Args:
        model: LogisticRegression instance
        X_train: sparse matrix of TF-IDF features
        y_train: training labels
    Returns:
        trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, target_names=None):
    """
    Evaluate model and print classification report.

    Args:
        model: trained model
        X: feature matrix
        y: true labels
        target_names: list of class names
    Returns:
        accuracy score
    """
    y_pred = model.predict(X)
    score = model.score(X, y)
    print(f"Accuracy: {score:.4f}")
    print(classification_report(y, y_pred,
          target_names=target_names))
    return score


def save_model(model, tfidf,
               model_path='data/processed/best_model.pkl',
               tfidf_path='data/processed/tfidf_vectorizer.pkl'):
    """
    Save trained model and vectorizer to disk.

    Args:
        model: trained model
        tfidf: fitted TfidfVectorizer
        model_path: path to save model
        tfidf_path: path to save vectorizer
    """
    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    print("Model and vectorizer saved!")


def load_model(model_path='data/processed/best_model.pkl',
               tfidf_path='data/processed/tfidf_vectorizer.pkl'):
    """
    Load trained model and vectorizer from disk.

    Args:
        model_path: path to saved model
        tfidf_path: path to saved vectorizer
    Returns:
        model, tfidf tuple
    """
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    return model, tfidf