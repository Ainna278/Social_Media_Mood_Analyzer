# src/vectorize.py

import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer():
    """
    Initialize TF-IDF vectorizer with project settings.
    
    Returns:
        TfidfVectorizer instance
    """
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

def fit_transform_train(tfidf, df_train):
    """
    Fit TF-IDF on train data and transform it.
    
    Args:
        tfidf: TfidfVectorizer instance
        df_train: training dataframe
    Returns:
        sparse matrix of TF-IDF features
    """
    return tfidf.fit_transform(df_train['cleaned_text'].fillna(''))

def transform_data(tfidf, df):
    """
    Transform data using fitted TF-IDF.
    
    Args:
        tfidf: fitted TfidfVectorizer instance
        df: dataframe to transform
    Returns:
        sparse matrix of TF-IDF features
    """
    return tfidf.transform(df['cleaned_text'].fillna(''))

def save_matrices(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save TF-IDF matrices and labels to disk.
    
    Args:
        X_train, X_val, X_test: sparse matrices
        y_train, y_val, y_test: label series
    """
    sp.save_npz('data/processed/X_train.npz', X_train)
    sp.save_npz('data/processed/X_val.npz', X_val)
    sp.save_npz('data/processed/X_test.npz', X_test)

    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_val.to_csv('data/processed/y_val.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    print("All matrices and labels saved!")