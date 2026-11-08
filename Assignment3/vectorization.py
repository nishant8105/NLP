# Text Vectorization for IMDB Reviews
# Run this file after preprocessing.py

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Folders
DATA = Path("data/processed")
MODELS = Path("models")
OUTPUT = Path("outputs/figures")

MODELS.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)


# Load training reviews
def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    return train["review"].fillna(""), test["review"].fillna("")


# Calculate percentage of zero values
def calculate_sparsity(matrix):
    total = matrix.shape[0] * matrix.shape[1]
    return 1 - (matrix.nnz / total)


# Create Bag-of-Words vectorizer
def create_bow(max_features=5000, min_df=2, max_df=1.0, binary=False,
               ngram_range=(1, 1)):
    return CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        binary=binary,
        ngram_range=ngram_range
    )


# Create TF-IDF vectorizer
def create_tfidf(max_features=5000, norm="l2", use_idf=True,
                 smooth_idf=True, sublinear_tf=False,
                 ngram_range=(1, 1)):
    return TfidfVectorizer(
        max_features=max_features,
        norm=norm,
        use_idf=use_idf,
        smooth_idf=smooth_idf,
        sublinear_tf=sublinear_tf,
        ngram_range=ngram_range
    )


# Task 2.1: Bag-of-Words experiments
def bow_experiments(X_train):
    results = []

    # Try different max_features, min_df, max_df and binary values
    experiments = [
        ("max_features", [1000, 5000, 10000]),
        ("min_df", [2, 5, 10]),
        ("max_df", [0.8, 0.9, 1.0]),
        ("binary", [True, False])
    ]

    for parameter, values in experiments:
        for value in values:
            settings = {
                "max_features": 5000,
                "min_df": 2,
                "max_df": 1.0,
                "binary": False
            }
            settings[parameter] = value

            vectorizer = create_bow(**settings)

            start = time.time()
            matrix = vectorizer.fit_transform(X_train)
            seconds = time.time() - start

            results.append({
                "parameter": parameter,
                "value": value,
                "vocabulary_size": len(vectorizer.vocabulary_),
                "sparsity": calculate_sparsity(matrix),
                "time_seconds": seconds
            })

    results = pd.DataFrame(results)
    results.to_csv(DATA / "bow_results.csv", index=False)

    # Plot vocabulary size
    results.plot(
        x="parameter", y="vocabulary_size", kind="bar",
        title="BoW Vocabulary Size", legend=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / "bow_vocabulary.png")
    plt.show()

    # Plot common words
    vectorizer = create_bow(max_features=20)
    matrix = vectorizer.fit_transform(X_train)
    counts = np.asarray(matrix.sum(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()

    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title("Common Words")
    plt.tight_layout()
    plt.savefig(OUTPUT / "bow_common_words.png")
    plt.show()

    # Save main BoW vectorizer
    main_vectorizer = create_bow()
    main_vectorizer.fit(X_train)
    joblib.dump(main_vectorizer, MODELS / "bow_vectorizer.pkl")

    return results


# Task 2.2: TF-IDF experiments
def tfidf_experiments(X_train):
    results = []

    # All combinations of the given TF-IDF settings
    settings = product(
        ["l1", "l2", None],
        [True, False],
        [True, False],
        [True, False]
    )

    for norm, use_idf, smooth_idf, sublinear_tf in settings:
        vectorizer = create_tfidf(
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf
        )

        start = time.time()
        matrix = vectorizer.fit_transform(X_train)
        seconds = time.time() - start

        results.append({
            "norm": str(norm),
            "use_idf": use_idf,
            "smooth_idf": smooth_idf,
            "sublinear_tf": sublinear_tf,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "sparsity": calculate_sparsity(matrix),
            "time_seconds": seconds
        })

    results = pd.DataFrame(results)
    results.to_csv(DATA / "tfidf_results.csv", index=False)

    # Fit the main TF-IDF vectorizer
    vectorizer = create_tfidf()
    matrix = vectorizer.fit_transform(X_train)

    # Find top 20 words
    average_scores = np.asarray(matrix.mean(axis=0)).flatten()
    words = vectorizer.get_feature_names_out()
    top_ids = average_scores.argsort()[-20:][::-1]

    top_words = pd.DataFrame({
        "word": words[top_ids],
        "score": average_scores[top_ids]
    })
    top_words.to_csv(DATA / "top_tfidf_words.csv", index=False)

    top_words.plot(
        x="word", y="score", kind="bar",
        title="Top 20 TF-IDF Words", legend=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / "top_tfidf_words.png")
    plt.show()

    # Plot TF-IDF score distribution
    plt.hist(matrix.data, bins=30)
    plt.title("TF-IDF Score Distribution")
    plt.xlabel("TF-IDF score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT / "tfidf_distribution.png")
    plt.show()

    # Plot scores for the first review
    sample = matrix[0].toarray().flatten()
    top_ids = sample.argsort()[-10:][::-1]

    plt.bar(words[top_ids], sample[top_ids])
    plt.xticks(rotation=45)
    plt.title("TF-IDF Scores for One Review")
    plt.tight_layout()
    plt.savefig(OUTPUT / "sample_tfidf.png")
    plt.show()

    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")

    return results, vectorizer


# Task 2.3: Word n-gram experiments
def ngram_experiments(X_train):
    ngrams = {
        "Unigram": (1, 1),
        "Bigram": (2, 2),
        "Trigram": (3, 3),
        "Unigram + Bigram": (1, 2),
        "Unigram + Trigram": (1, 3),
        "Bigram + Trigram": (2, 3),
        "All": (1, 3)
    }

    results = []

    for name, ngram_range in ngrams.items():
        vectorizer = create_bow(
            max_features=10000,
            ngram_range=ngram_range
        )

        start = time.time()
        matrix = vectorizer.fit_transform(X_train)
        seconds = time.time() - start

        results.append({
            "ngram": name,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "sparsity": calculate_sparsity(matrix),
            "time_seconds": seconds
        })

        # Print common bigrams and trigrams
        if name in ["Bigram", "Trigram"]:
            counts = np.asarray(matrix.sum(axis=0)).flatten()
            words = vectorizer.get_feature_names_out()
            top_ids = counts.argsort()[-10:][::-1]
            print("\\nTop", name, "features:")
            print(words[top_ids])

    results = pd.DataFrame(results)
    results.to_csv(DATA / "ngram_results.csv", index=False)

    results.plot(
        x="ngram", y="vocabulary_size", kind="bar",
        title="N-gram Vocabulary Size", legend=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / "ngram_vocabulary.png")
    plt.show()

    return results


# Task 2.4: Character n-grams
def character_ngrams(X_train):
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        max_features=10000
    )

    start = time.time()
    matrix = vectorizer.fit_transform(X_train)
    seconds = time.time() - start

    print("\nCharacter n-grams")
    print("Vocabulary size:", len(vectorizer.vocabulary_))
    print("Matrix shape:", matrix.shape)
    print("Sparsity:", calculate_sparsity(matrix))
    print("Time taken:", round(seconds, 2), "seconds")

    joblib.dump(vectorizer, MODELS / "char_vectorizer.pkl")

    return vectorizer

