from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------------
# 1. Create TF-IDF features
# ---------------------------------------------------------

def create_tfidf_features(x_train, x_val, x_test,
                          max_features=30000, ngram_range=(1, 2)):
    """
    Fit TF-IDF on training text only, then transform validation and test text.
    Returns the vectorizer and the three feature matrices.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        stop_words=None
    )

    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_val_tfidf = vectorizer.transform(x_val)
    x_test_tfidf = vectorizer.transform(x_test)

    print("TF-IDF vocabulary size:", len(vectorizer.vocabulary_))
    print("Training feature shape:", x_train_tfidf.shape)

    return vectorizer, x_train_tfidf, x_val_tfidf, x_test_tfidf


# ---------------------------------------------------------
# 2. Evaluation helper
# ---------------------------------------------------------

def calculate_multilabel_metrics(y_true, y_pred, model_name):
    """Calculate common multi-label classification metrics."""
    results = {
        "model": model_name,
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        )
    }
    return results


# ---------------------------------------------------------
# 3. Logistic Regression with TF-IDF
# ---------------------------------------------------------

def train_tfidf_logistic_regression(
    x_train_tfidf,
    y_train,
    x_test_tfidf,
    y_test,
    max_iter=1000
):
    """
    Train one-vs-rest Logistic Regression for multi-label classification.
    Returns the trained model, predictions and metrics.
    """
    model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="liblinear"
        )
    )

    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)

    metrics = calculate_multilabel_metrics(
        y_test, predictions, "TF-IDF Logistic Regression"
    )

    print(pd.Series(metrics))
    return model, predictions, metrics


# ---------------------------------------------------------
# 4. Random Forest with TF-IDF
# ---------------------------------------------------------

def train_tfidf_random_forest(
    x_train_tfidf,
    y_train,
    x_test_tfidf,
    y_test,
    n_estimators=100,
    max_depth=None
):
    """
    Train Random Forest for multi-label classification.
    Random Forest accepts a multi-label indicator target matrix.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)

    metrics = calculate_multilabel_metrics(
        y_test, predictions, "TF-IDF Random Forest"
    )

    print(pd.Series(metrics))
    return model, predictions, metrics


# ---------------------------------------------------------
# 5. Build a Word2Vec embedding matrix
# ---------------------------------------------------------

def create_word2vec_embedding_matrix(
    tokenizer,
    word2vec_model,
    embedding_dim=None,
    output_path="models/word2vec_emotion_embedding_matrix.npy"
):
    """
    Create an embedding matrix using the same word indexes as the Keras
    tokenizer. Words missing from Word2Vec remain as zero vectors.
    """
    word_index = tokenizer.word_index

    if embedding_dim is None:
        embedding_dim = word2vec_model.vector_size

    matrix = np.zeros(
        (len(word_index) + 1, embedding_dim),
        dtype=np.float32
    )

    found = 0
    for word, index in word_index.items():
        if word in word2vec_model.wv:
            vector = word2vec_model.wv[word]
            if len(vector) == embedding_dim:
                matrix[index] = vector
                found += 1

    coverage = found / len(word_index) * 100 if word_index else 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)

    print("Word2Vec embedding matrix shape:", matrix.shape)
    print(f"Word2Vec vocabulary coverage: {coverage:.2f}%")

    return matrix, coverage


# ---------------------------------------------------------
# 6. Build and train a simple embedding classifier
# ---------------------------------------------------------

def build_embedding_classifier(
    embedding_matrix,
    max_length=100,
    num_classes=28,
    trainable=False
):
    """
    Build a simple neural network that can use either Word2Vec or GloVe.
    Uses sigmoid for multi-label emotion classification.
    """
    model = Sequential([
        Input(shape=(max_length,)),
        Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable
        ),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_embedding_classifier(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=30,
    batch_size=32,
    sample_weight=None
):
    """Train the embedding classifier with early stopping."""
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        sample_weight=sample_weight
    )

    return history


def evaluate_embedding_classifier(model, x_test, y_test, model_name):
    """Predict test labels and calculate multi-label metrics."""
    probabilities = model.predict(x_test)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = calculate_multilabel_metrics(
        y_test, predictions, model_name
    )

    print(pd.Series(metrics))
    return predictions, probabilities, metrics


# ---------------------------------------------------------
# 7. Compare all models in one table
# ---------------------------------------------------------

def compare_model_results(results, output_path="outputs/tables/model_comparison.csv"):
    """
    Combine metric dictionaries into a comparison table.
    Pass a list such as [logistic_metrics, rf_metrics, glove_metrics].
    """
    df = pd.DataFrame(results)

    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Comparison table saved to:", output_path)
    return df


def plot_model_comparison(results_df,
                          output_path="outputs/figures/model_comparison.png"):
    """Plot macro F1 for each model."""
    if "model" not in results_df.columns or "macro_f1" not in results_df.columns:
        raise ValueError("Results must contain 'model' and 'macro_f1' columns.")

    results_df = results_df.sort_values("macro_f1")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.barh(results_df["model"], results_df["macro_f1"])
    plt.xlabel("Macro F1-score")
    plt.ylabel("Model")
    plt.title("Model Comparison")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


# ---------------------------------------------------------
# 8. Save trained baseline models
# ---------------------------------------------------------

def save_baseline_model(model, output_path):
    """Save a scikit-learn model with joblib."""
    import joblib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print("Model saved to:", output_path)


def save_tfidf_vectorizer(vectorizer,
                          output_path="models/tfidf_vectorizer.joblib"):
    """Save the fitted TF-IDF vectorizer."""
    import joblib

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_path)
    print("Vectorizer saved to:", output_path)
