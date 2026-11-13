import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Conv1D,
    GlobalMaxPooling1D,
    Bidirectional,
    LSTM
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------
# 1. Load GoEmotions dataset
# ---------------------------------------------------------

def load_goemotions(data_path, text_column="text", label_column="labels"):
    """
    Load CSV files from a folder or one CSV file.

    Supported formats:
    - text + labels columns, where labels may be a list/string
    - text + separate emotion columns containing 0/1 values

    Returns:
        dataframe, list_of_emotion_names, multi_label_array
    """
    data_path = Path(data_path)

    if data_path.is_file():
        files = [data_path]
    else:
        files = list(data_path.rglob("*.csv"))

    if not files:
        raise FileNotFoundError("No CSV files found in " + str(data_path))

    frames = []
    for file in files:
        print("Reading:", file)
        frames.append(pd.read_csv(file))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[text_column]).copy()
    df[text_column] = df[text_column].astype(str)

    # Format 1: separate column for every emotion
    known_non_emotion_columns = {
        text_column, label_column, "id", "author", "subreddit",
        "link_id", "parent_id", "created_utc", "rater_id",
        "example_very_unclear"
    }

    possible_emotion_columns = [
        col for col in df.columns
        if col not in known_non_emotion_columns
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if possible_emotion_columns:
        emotion_columns = possible_emotion_columns
        labels = df[emotion_columns].fillna(0).astype(int).values
        # Keep only rows that have at least one emotion label.
        keep = labels.sum(axis=1) > 0
        df = df.loc[keep].reset_index(drop=True)
        labels = labels[keep]
        emotion_names = list(emotion_columns)

    # Format 2: one column contains label names
    elif label_column in df.columns:
        label_values = df[label_column].fillna("").astype(str).tolist()

        parsed_labels = []
        emotion_names = set()

        for value in label_values:
            # Handles strings such as "['joy', 'love']" or "joy, love".
            value = value.strip()
            value = value.replace("[", "").replace("]", "")
            value = value.replace("'", "").replace('"', "")
            parts = [item.strip() for item in re.split(r"[,;|]", value)]
            parts = [item for item in parts if item and item.lower() != "nan"]
            parsed_labels.append(parts)
            emotion_names.update(parts)

        emotion_names = sorted(emotion_names)
        label_to_index = {name: i for i, name in enumerate(emotion_names)}

        labels = np.zeros((len(df), len(emotion_names)), dtype=np.float32)
        for row_index, row_labels in enumerate(parsed_labels):
            for label in row_labels:
                labels[row_index, label_to_index[label]] = 1

        keep = labels.sum(axis=1) > 0
        df = df.loc[keep].reset_index(drop=True)
        labels = labels[keep]

    else:
        raise ValueError(
            "Could not find emotion columns. Check text_column and label_column."
        )

    print("Rows loaded:", len(df))
    print("Number of emotion labels:", len(emotion_names))
    print("Emotion names:", emotion_names)

    return df, emotion_names, labels.astype(np.float32)


# ---------------------------------------------------------
# 2. Explore the dataset
# ---------------------------------------------------------

def plot_emotion_distribution(labels, emotion_names,
                              output_path="outputs/figures/emotion_distribution.png"):
    """Plot the number of comments for each emotion."""
    counts = labels.sum(axis=0)
    order = np.argsort(counts)

    plt.figure(figsize=(12, 8))
    plt.barh(
        [emotion_names[i] for i in order],
        counts[order]
    )
    plt.xlabel("Number of comments")
    plt.ylabel("Emotion")
    plt.title("GoEmotions Class Distribution")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def get_emotion_counts(labels, emotion_names):
    """Return a table of emotion counts and percentages."""
    counts = labels.sum(axis=0)
    result = pd.DataFrame({
        "emotion": emotion_names,
        "count": counts.astype(int),
        "percentage": counts / len(labels) * 100
    })
    return result.sort_values("count")


def plot_comment_lengths(texts,
                         output_path="outputs/figures/comment_length_distribution.png"):
    """Plot the number of words in each comment."""
    lengths = [len(str(text).split()) for text in texts]

    plt.figure(figsize=(9, 5))
    plt.hist(lengths, bins=40)
    plt.xlabel("Number of words")
    plt.ylabel("Number of comments")
    plt.title("Comment Length Distribution")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    return pd.Series(lengths).describe()


def show_emotion_examples(texts, labels, emotion_names, number=5):
    """Print up to five example comments for every emotion."""
    for emotion_index, emotion in enumerate(emotion_names):
        print("\n", emotion.upper())
        indexes = np.where(labels[:, emotion_index] == 1)[0][:number]

        for index in indexes:
            print("-", texts.iloc[index] if hasattr(texts, "iloc") else texts[index])


# ---------------------------------------------------------
# 3. Clean text and split the data
# ---------------------------------------------------------

def clean_emotion_text(text):
    """
    Clean a comment.
    Keep 'not' and 'no' because they can change the emotion meaning.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_emotion_texts(texts):
    """Clean all comments and return a list."""
    return [clean_emotion_text(text) for text in texts]


def split_data(texts, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split data into 70% train, 15% validation and 15% test.

    Stratification is based on each comment's first active label.
    This is a simple approach for multi-label data; it does not guarantee
    that every rare emotion appears in every split.
    """
    texts = np.array(texts)
    labels = np.array(labels)

    # Use the first active emotion as a stratification label.
    main_labels = np.argmax(labels, axis=1)

    x_train, x_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=main_labels
    )

    temp_main_labels = np.argmax(y_temp, axis=1)
    relative_test_size = test_size / (test_size + val_size)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_main_labels
    )

    print("Training samples:", len(x_train))
    print("Validation samples:", len(x_val))
    print("Testing samples:", len(x_test))

    return x_train, x_val, x_test, y_train, y_val, y_test


# ---------------------------------------------------------
# 4. Tokenize comments and create sequences
# ---------------------------------------------------------

def prepare_tokenizer(x_train, num_words=30000, oov_token="<OOV>"):
    """Fit the tokenizer on training text only."""
    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token=oov_token
    )
    tokenizer.fit_on_texts(x_train)
    return tokenizer


def convert_to_sequences(tokenizer, texts, max_length=100):
    """Convert comments into padded integer sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )
    return padded


def save_tokenizer_vocabulary(tokenizer, output_path="models/tokenizer_word_index.json"):
    """Save the tokenizer word index."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(tokenizer.word_index, file, indent=2)


# ---------------------------------------------------------
# 5. Load GloVe vectors and check coverage
# ---------------------------------------------------------

def load_glove(filepath):
    """Load a Stanford GloVe text file into a dictionary."""
    embeddings = {}

    with open(filepath, "r", encoding="utf-8") as file:
        for line in file:
            values = line.strip().split()
            if len(values) < 2:
                continue

            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector

    print("GloVe words loaded:", len(embeddings))
    return embeddings


def check_glove_coverage(tokenizer, embeddings):
    """Check how many tokenizer words are present in GloVe."""
    words = tokenizer.word_index
    found = [word for word in words if word in embeddings]

    total_words = len(words)
    found_words = len(found)
    coverage = (found_words / total_words * 100) if total_words else 0

    print("Words in dataset vocabulary:", total_words)
    print("Words found in GloVe:", found_words)
    print(f"Vocabulary coverage: {coverage:.2f}%")

    return {
        "total_words": total_words,
        "found_words": found_words,
        "coverage_percentage": coverage
    }


def create_embedding_matrix(tokenizer, embeddings, embedding_dim=100,
                            output_path="models/glove_embedding_matrix.npy"):
    """Create a matrix that matches the tokenizer word indexes."""
    word_index = tokenizer.word_index
    matrix = np.zeros((len(word_index) + 1, embedding_dim), dtype=np.float32)

    found = 0
    for word, index in word_index.items():
        vector = embeddings.get(word)

        if vector is not None and len(vector) == embedding_dim:
            matrix[index] = vector
            found += 1

    coverage = found / len(word_index) * 100 if word_index else 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)

    print("Embedding matrix shape:", matrix.shape)
    print(f"Embedding matrix coverage: {coverage:.2f}%")

    return matrix, coverage


# ---------------------------------------------------------
# 6. Create neural network models
# ---------------------------------------------------------

def build_dense_model(embedding_matrix, max_length=100,
                      trainable=False, num_classes=28):
    """Dense network with Global Average Pooling."""
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
    return model


def build_cnn_model(embedding_matrix, max_length=100,
                    trainable=False, num_classes=28):
    """CNN model for text classification."""
    model = Sequential([
        Input(shape=(max_length,)),
        Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable
        ),
        Conv1D(128, kernel_size=5, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="sigmoid")
    ])
    return model


def build_lstm_model(embedding_matrix, max_length=100,
                     trainable=False, num_classes=28):
    """Bidirectional LSTM model for text classification."""
    model = Sequential([
        Input(shape=(max_length,)),
        Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=trainable
        ),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="sigmoid")
    ])
    return model


def compile_model(model, learning_rate=0.001, multi_label=True):
    """Compile a model for multi-label or single-label classification."""
    if multi_label:
        loss = "binary_crossentropy"
    else:
        loss = "categorical_crossentropy"

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------
# 7. Class weights and model training
# ---------------------------------------------------------

def calculate_class_weights(y_train):
    """
    Calculate a weight for every emotion.
    These weights can be used to create sample weights for multi-label data.
    """
    counts = np.sum(y_train, axis=0)
    total = len(y_train)
    number_of_classes = y_train.shape[1]

    weights = total / (number_of_classes * np.maximum(counts, 1))
    return weights.astype(np.float32)


def create_sample_weights(y_train):
    """
    Give each training comment a weight based on its rarest active emotion.
    This is a simple way to account for imbalance in multi-label data.
    """
    class_weights = calculate_class_weights(y_train)
    sample_weights = []

    for row in y_train:
        active_weights = class_weights[row == 1]
        if len(active_weights):
            sample_weights.append(float(np.max(active_weights)))
        else:
            sample_weights.append(1.0)

    return np.asarray(sample_weights, dtype=np.float32)


def train_model(model, x_train, y_train, x_val, y_val,
                epochs=30, batch_size=32, sample_weight=None):
    """Train a model with early stopping."""
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


def save_classifier(model, model_path):
    """Save a trained Keras model."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print("Model saved to:", model_path)


def plot_training_history(history, model_name="model",
                          output_dir="outputs/figures"):
    """Save loss and accuracy curves."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_name + " Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss.png", dpi=300)
    plt.show()
    plt.close()

    if "accuracy" in history.history:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["accuracy"], label="Training accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(model_name + " Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_accuracy.png", dpi=300)
        plt.show()
        plt.close()


# ---------------------------------------------------------
# 8. Multi-label evaluation
# ---------------------------------------------------------

def evaluate_multilabel_model(model, x_test, y_test,
                              threshold=0.5, top_k=3):
    """Calculate F1, Hamming loss, precision@k and recall@k."""
    probabilities = model.predict(x_test)
    predictions = (probabilities >= threshold).astype(int)

    micro_f1 = f1_score(y_test, predictions, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)
    ham_loss = hamming_loss(y_test, predictions)

    precision_values = []
    recall_values = []

    for true_row, score_row in zip(y_test, probabilities):
        selected = np.argsort(score_row)[-top_k:]
        predicted_row = np.zeros(len(score_row), dtype=int)
        predicted_row[selected] = 1

        true_count = np.sum(true_row)
        correct = np.sum(predicted_row * true_row)

        precision_values.append(correct / top_k)
        recall_values.append(correct / true_count if true_count else 0)

    results = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hamming_loss": ham_loss,
        f"precision_at_{top_k}": float(np.mean(precision_values)),
        f"recall_at_{top_k}": float(np.mean(recall_values))
    }

    print(pd.Series(results))
    return results, probabilities, predictions


def save_evaluation_results(results, output_path="outputs/tables/model_metrics.csv",
                            model_name="model"):
    """Save evaluation metrics to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"model": model_name, **results}
    df = pd.DataFrame([row])

    if output_path.exists():
        old_df = pd.read_csv(output_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(output_path, index=False)
    return df


def save_classification_report(y_true, y_pred, emotion_names,
                               output_path="outputs/tables/classification_report.csv"):
    """Save precision, recall and F1 for each emotion."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=emotion_names,
        output_dict=True,
        zero_division=0
    )

    df = pd.DataFrame(report).transpose()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)
    return df


# ---------------------------------------------------------
# 9. Optional single-label conversion for softmax models
# ---------------------------------------------------------

def convert_to_single_label(labels):
    """
    Convert multi-label rows to one class using the first active label.
    Use only if your assignment experiment specifically requires softmax.
    This loses additional labels, so multi-label sigmoid is preferred
    for the full GoEmotions task.
    """
    class_ids = np.argmax(labels, axis=1)
    return tf.keras.utils.to_categorical(
        class_ids,
        num_classes=labels.shape[1]
    )
