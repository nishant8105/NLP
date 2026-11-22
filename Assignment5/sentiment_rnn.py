import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, LSTM, GRU, Bidirectional, Dropout, Dense
from keras.callbacks import EarlyStopping


# Create folders used in this project
os.makedirs("./data/processed", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./outputs/tables", exist_ok=True)
os.makedirs("./outputs/figures", exist_ok=True)


def clean_text(text):
    """Clean review text and keep words such as not, no and never."""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_prepare_data(max_words=10000, max_length=200):
    """Load the IMDB CSV file, prepare text and split the data."""

    # Read the dataset from the raw data folder
    data = pd.read_csv("./data/raw/IMDB Dataset.csv")

    # This dataset should contain review and sentiment columns
    data = data[["review", "sentiment"]].dropna()

    # Convert sentiment labels to numbers: positive = 1, negative = 0
    data["sentiment"] = data["sentiment"].map({
        "positive": 1,
        "negative": 0
    })
    data = data.dropna(subset=["sentiment"])

    # Clean all review text
    reviews = [clean_text(str(review)) for review in data["review"]]
    labels = data["sentiment"].astype(int).values

    # Split into 80% training and 20% testing
    np.random.seed(42)
    indexes = np.random.permutation(len(reviews))
    split = int(len(indexes) * 0.8)

    train_indexes = indexes[:split]
    test_indexes = indexes[split:]

    train_reviews = [reviews[i] for i in train_indexes]
    test_reviews = [reviews[i] for i in test_indexes]
    y_train_all = labels[train_indexes]
    y_test = labels[test_indexes]

    # Create tokenizer using training reviews only
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_reviews)

    x_train_all = tokenizer.texts_to_sequences(train_reviews)
    x_test = tokenizer.texts_to_sequences(test_reviews)

    # Save original review lengths
    lengths = [len(review) for review in x_train_all]

    # Pad reviews to the same length
    x_train_all = pad_sequences(
        x_train_all, maxlen=max_length, padding="post", truncating="post"
    )
    x_test = pad_sequences(
        x_test, maxlen=max_length, padding="post", truncating="post"
    )

    # Split training data into 80% training and 20% validation
    indexes = np.random.permutation(len(x_train_all))
    split = int(len(indexes) * 0.8)

    train_indexes = indexes[:split]
    val_indexes = indexes[split:]

    x_val = x_train_all[val_indexes]
    y_val = y_train_all[val_indexes]
    x_train = x_train_all[train_indexes]
    y_train = y_train_all[train_indexes]

    vocab_size = min(max_words, len(tokenizer.word_index) + 1)

    # Save processed data and tokenizer
    np.savez(
        "./data/processed/imdb_data.npz",
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
        sequence_lengths=np.array(lengths)
    )

    with open("./data/processed/tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    return x_train, y_train, x_val, y_val, x_test, y_test, vocab_size, lengths


def plot_sequence_lengths(lengths, max_length=200):
    """Plot and save the review length graph."""
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=50)
    plt.axvline(max_length, color="red", linestyle="--", label="Selected length")
    plt.xlabel("Number of words")
    plt.ylabel("Number of reviews")
    plt.title("IMDB Review Length Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./outputs/figures/sequence_length.png")
    plt.show()

    summary = pd.Series(lengths).describe()
    summary.to_csv("./outputs/tables/sequence_length_summary.csv")
    return summary


def create_model(model_type, vocab_size, max_length, units=64, dropout=0.5, layers=2):
    """Create a Simple RNN, LSTM, BiLSTM or GRU model."""

    model = Sequential()
    model.add(Embedding(vocab_size, 128, input_length=max_length))

    if model_type == "rnn":
        model.add(SimpleRNN(units))
        model.add(Dropout(dropout))

    elif model_type == "lstm":
        for i in range(layers):
            if i == layers - 1:
                model.add(LSTM(units))
            else:
                model.add(LSTM(units, return_sequences=True))
            model.add(Dropout(dropout))

    elif model_type == "bilstm":
        model.add(Bidirectional(LSTM(units, return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(units // 2)))
        model.add(Dropout(dropout))

    elif model_type == "gru":
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(dropout))
        model.add(GRU(units // 2))
        model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=10, early_stop=False):
    """Train a model and return its history and training time."""

    callbacks = []
    if early_stop:
        callbacks.append(EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        ))

    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks
    )
    training_time = time.time() - start
    return history, training_time


def evaluate_model(model, x_test, y_test):
    """Evaluate a model on the test data."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return loss, accuracy


def save_model(model, name):
    """Save a trained model."""
    model.save("./models/" + name + ".keras")


def plot_training(history, name):
    """Save accuracy and loss graphs."""
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Training")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title(name + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./outputs/figures/" + name.lower() + "_accuracy.png")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Training")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.title(name + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./outputs/figures/" + name.lower() + "_loss.png")
    plt.show()


def save_results(results):
    """Save model comparison results as a CSV table."""
    table = pd.DataFrame(results)
    table.to_csv("./outputs/tables/model_comparison.csv", index=False)
    return table
