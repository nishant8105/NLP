# Feature Engineering and Feature Selection for IMDB Reviews

import re
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


DATA = Path("data/processed")
MODELS = Path("models")
OUTPUT = Path("outputs/figures")

MODELS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


# Simple positive and negative word lists
POSITIVE_WORDS = [
    "good", "great", "excellent", "amazing", "wonderful",
    "love", "loved", "best", "enjoy", "enjoyed", "perfect"
]

NEGATIVE_WORDS = [
    "bad", "terrible", "awful", "worst", "boring",
    "hate", "hated", "poor", "waste", "disappointing"
]

KEYWORDS = ["amazing", "terrible", "worst"]


# Load training and testing data
def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    X_train = train["review"].fillna("").astype(str)
    X_test = test["review"].fillna("").astype(str)
    y_train = train["sentiment"]
    y_test = test["sentiment"]

    # If raw_review is available, use it for custom features.
    # Otherwise, use the processed review.
    raw_train = train["raw_review"] if "raw_review" in train.columns else X_train
    raw_test = test["raw_review"] if "raw_review" in test.columns else X_test

    return X_train, X_test, raw_train, raw_test, y_train, y_test


# Task 4.1: Create custom features
def extract_features(reviews):
    rows = []

    for review in reviews:
        text = str(review)
        words = re.findall(r"[a-zA-Z']+", text.lower())
        word_count = len(words)

        # Sentiment lexicon features
        positive_count = sum(word in POSITIVE_WORDS for word in words)
        negative_count = sum(word in NEGATIVE_WORDS for word in words)

        # Statistical features
        average_word_length = (
            sum(len(word) for word in words) / max(1, word_count)
        )

        sentence_count = len(re.findall(r"[.!?]+", text))
        if sentence_count == 0:
            sentence_count = 1

        capital_ratio = (
            sum(char.isupper() for char in text) / max(1, len(text))
        )

        # Domain-specific features
        row = {
            "character_count": len(text),
            "word_count": word_count,
            "average_word_length": average_word_length,
            "sentence_count": sentence_count,
            "exclamation_count": text.count("!"),
            "capital_ratio": capital_ratio,
            "positive_word_count": positive_count,
            "negative_word_count": negative_count,
            "polarity_ratio": (
                (positive_count - negative_count) / max(1, word_count)
            )
        }

        for word in KEYWORDS:
            row["has_" + word] = int(word in words)

        # Count simple rating expressions, such as 8/10 or 5 stars.
        row["rating_mention"] = int(
            bool(re.search(r"\b(10|[1-9])/10\b|\b[1-5]\s*stars?\b", text.lower()))
        )

        rows.append(row)

    return pd.DataFrame(rows).fillna(0)


# Combine text features and custom features
def combine_features(X_train, X_test, raw_train, raw_test):
    tfidf = TfidfVectorizer(max_features=5000)
    train_text = tfidf.fit_transform(X_train)
    test_text = tfidf.transform(X_test)

    train_custom = extract_features(raw_train)
    test_custom = extract_features(raw_test)

    scaler = StandardScaler()
    train_custom = scaler.fit_transform(train_custom)
    test_custom = scaler.transform(test_custom)

    train_combined = hstack([
        train_text, csr_matrix(train_custom)
    ]).tocsr()

    test_combined = hstack([
        test_text, csr_matrix(test_custom)
    ]).tocsr()

    joblib.dump(tfidf, MODELS / "feature_tfidf.pkl")
    joblib.dump(scaler, MODELS / "custom_feature_scaler.pkl")

    return train_text, test_text, train_combined, test_combined


# Task 4.2: Feature selection using Chi-square
def select_chi_square(X_train, y_train, X_test, k=1000):
    selector = SelectKBest(score_func=chi2, k=min(k, X_train.shape[1]))
    train_selected = selector.fit_transform(X_train, y_train)
    test_selected = selector.transform(X_test)

    joblib.dump(selector, MODELS / "chi_square_selector.pkl")
    return train_selected, test_selected

# Mutual Information score function
def calculate_mutual_information(X, y):
    return mutual_info_classif(
        X,
        y,
        discrete_features=True,
        random_state=42
    )
# Feature selection using Mutual Information
def select_mutual_information(X_train, y_train, X_test, k=1000):

    selector = SelectKBest(
        score_func=calculate_mutual_information,
        k=min(k, X_train.shape[1])
    )

    train_selected = selector.fit_transform(X_train, y_train)
    test_selected = selector.transform(X_test)

    joblib.dump(
        selector,
        MODELS / "mutual_info_selector.pkl"
    )

    return train_selected, test_selected

# Feature selection using L1 regularization
def select_l1_features(X_train, y_train, X_test):
    l1_model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=0.1,
        max_iter=1000
    )

    selector = SelectFromModel(l1_model)
    train_selected = selector.fit_transform(X_train, y_train)
    test_selected = selector.transform(X_test)

    joblib.dump(selector, MODELS / "l1_selector.pkl")
    return train_selected, test_selected


# Train and measure one model
def train_and_measure(name, model, X_train, y_train, X_test, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    start = time.time()
    predictions = model.predict(X_test)
    inference_time = time.time() - start

    return {
        "Vectorization / Features": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "F1-Score": f1_score(y_test, predictions, average="weighted"),
        "Training Time (seconds)": training_time,
        "Inference Time (seconds)": inference_time
    }


# Compare models with and without feature selection
def compare_features(X_train, y_train, X_test, y_test):
    results = []

    # Compare BoW + Naive Bayes
    bow = CountVectorizer(max_features=5000)
    bow_train = bow.fit_transform(X_train)
    bow_test = bow.transform(X_test)

    results.append(train_and_measure(
        "BoW (5k) + Multinomial NB",
        MultinomialNB(),
        bow_train, y_train, bow_test, y_test
    ))

    # Compare TF-IDF + Logistic Regression
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_train = tfidf.fit_transform(X_train)
    tfidf_test = tfidf.transform(X_test)

    results.append(train_and_measure(
        "TF-IDF (5k) + Logistic Regression",
        LogisticRegression(max_iter=1000),
        tfidf_train, y_train, tfidf_test, y_test
    ))

    # Chi-square selected features
    train_chi, test_chi = select_chi_square(
        tfidf_train, y_train, tfidf_test
    )
    results.append(train_and_measure(
        "TF-IDF + Chi-square",
        LogisticRegression(max_iter=1000),
        train_chi, y_train, test_chi, y_test
    ))

    # Mutual information selected features
    train_mi, test_mi = select_mutual_information(
        tfidf_train, y_train, tfidf_test
    )
    results.append(train_and_measure(
        "TF-IDF + Mutual Information",
        LogisticRegression(max_iter=1000),
        train_mi, y_train, test_mi, y_test
    ))

    # L1 selected features
    train_l1, test_l1 = select_l1_features(
        tfidf_train, y_train, tfidf_test
    )
    results.append(train_and_measure(
        "TF-IDF + L1 Selection",
        LogisticRegression(max_iter=1000),
        train_l1, y_train, test_l1, y_test
    ))

    return pd.DataFrame(results)


# Compare text-only features with text + custom features
def compare_custom_features(train_text, test_text, train_combined,
                            test_combined, y_train, y_test):
    results = []

    results.append(train_and_measure(
        "TF-IDF only",
        LogisticRegression(max_iter=1000),
        train_text, y_train, test_text, y_test
    ))

    results.append(train_and_measure(
        "TF-IDF + Custom Features",
        LogisticRegression(max_iter=1000),
        train_combined, y_train, test_combined, y_test
    ))

    return pd.DataFrame(results)


# Save comparison table and chart
def save_comparison(results, filename):
    results.to_csv(DATA / filename, index=False)

    print(results)

    results.plot(
        x="Vectorization / Features",
        y=["Accuracy", "F1-Score"],
        kind="bar",
        figsize=(10, 5)
    )
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT / (filename.replace(".csv", ".png")))
    plt.show()

