import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.callbacks import Callback

from sampling_strategies import choose_next_character


# Create project folders
os.makedirs("./data/processed", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./outputs/tables", exist_ok=True)
os.makedirs("./outputs/figures", exist_ok=True)


def load_text_corpus():
    """Load review text from the IMDB CSV and save it for text generation."""

    # Read the same raw dataset used in the sentiment analysis task
    data = pd.read_csv("./data/raw/IMDB Dataset.csv")
    data = data["review"].dropna().astype(str)

    # Join reviews into one long text corpus
    text = "\n".join(data.tolist())
    text = text.lower()

    # Save the corpus so it can be reused
    with open("./data/processed/text_generation_corpus.txt", "w", encoding="utf-8") as file:
        file.write(text)

    return text


def prepare_character_data(text, seq_length=100):
    """Create character mappings and input/next-character training sequences."""

    # Find all different characters
    chars = sorted(list(set(text)))
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = {i: char for char, i in char_to_index.items()}

    # Convert text characters into numbers
    encoded_text = np.array([char_to_index[char] for char in text], dtype=np.int32)

    # Create sequences and the next character after each sequence
    sequences = []
    next_chars = []

    for i in range(len(encoded_text) - seq_length):
        sequences.append(encoded_text[i:i + seq_length])
        next_chars.append(encoded_text[i + seq_length])

    x = np.array(sequences, dtype=np.int32)
    y = np.array(next_chars, dtype=np.int32)

    # Save processed character data
    np.savez(
        "./data/processed/character_sequences.npz",
        x=x, y=y
    )

    # Save character mappings
    with open("./data/processed/character_mapping.json", "w", encoding="utf-8") as file:
        json.dump(char_to_index, file, ensure_ascii=False, indent=2)

    # Save character frequency table
    frequencies = pd.Series(list(text)).value_counts().rename_axis("character").reset_index(name="count")
    frequencies.to_csv("./outputs/tables/character_frequencies.csv", index=False)

    return x, y, char_to_index, index_to_char, chars, frequencies


def build_character_model(vocab_size, seq_length=100):
    """Build a character-level LSTM model using integer character inputs."""

    model = Sequential()
    model.add(Embedding(vocab_size, 64, input_length=seq_length))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam"
    )

    return model


class SaveEveryFiveEpochs(Callback):
    """Save a model checkpoint every five epochs and generate a sample."""

    def __init__(self, index_to_char, seed_text, seq_length, temperature=1.0):
        super().__init__()
        self.index_to_char = index_to_char
        self.seed_text = seed_text
        self.seq_length = seq_length
        self.temperature = temperature

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            self.model.save(f"./models/text_generation_epoch_{epoch + 1}.keras")
            print("\nSample after epoch", epoch + 1)
            print(generate_text(
                self.model, self.seed_text, self.index_to_char,
                self.seq_length, length=150, temperature=self.temperature
            ))


def train_character_model(model, x, y, index_to_char, seq_length=100,
                          epochs=30, batch_size=128, seed_text="the"):
    """Train the character model and save its training history and final model."""

    callback = SaveEveryFiveEpochs(
        index_to_char, seed_text, seq_length
    )

    history = model.fit(
        x, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[callback],
        verbose=1
    )

    model.save("./models/character_lstm.keras")

    # Save loss values in a table
    loss_table = pd.DataFrame({
        "epoch": range(1, len(history.history["loss"]) + 1),
        "loss": history.history["loss"]
    })
    loss_table.to_csv("./outputs/tables/text_generation_loss.csv", index=False)

    # Save training loss graph
    plt.figure(figsize=(8, 4))
    plt.plot(loss_table["epoch"], loss_table["loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Character LSTM Training Loss")
    plt.tight_layout()
    plt.savefig("./outputs/figures/text_generation_loss.png")
    plt.show()

    return history


def generate_text(model, seed_text, index_to_char, seq_length=100,
                  length=500, temperature=1.0, strategy="temperature",
                  top_k=5, top_p=0.9):
    """Generate text by using functions from sampling_strategies.py."""

    char_to_index = {char: index for index, char in index_to_char.items()}
    generated = seed_text.lower()

    for i in range(length):
        # Take the last seq_length characters
        seed = generated[-seq_length:]

        # Add spaces if the seed is shorter than the required length
        if len(seed) < seq_length:
            seed = seed.rjust(seq_length)

        # Convert characters into numbers
        x = np.array([
            char_to_index.get(char, 0) for char in seed
        ]).reshape(1, seq_length)

        # Predict the next character probabilities
        probabilities = model.predict(x, verbose=0)[0]

        # Use the selected sampling function
        next_index = choose_next_character(
            probabilities,
            strategy=strategy,
            temperature=temperature,
            k=top_k,
            p=top_p
        )

        generated += index_to_char[next_index]

    return generated


def generate_samples(model, seed_texts, index_to_char, seq_length=100):
    """Generate 500-character samples for several seeds and sampling settings."""

    results = []

    for seed in seed_texts:
        # Greedy sampling
        results.append({
            "seed": seed,
            "strategy": "greedy",
            "temperature": None,
            "generated_text": generate_text(
                model, seed, index_to_char, seq_length,
                length=500, strategy="greedy"
            )
        })

        # Three temperature values
        for temp in [0.5, 1.0, 1.5]:
            results.append({
                "seed": seed,
                "strategy": "temperature",
                "temperature": temp,
                "generated_text": generate_text(
                    model, seed, index_to_char, seq_length,
                    length=500, temperature=temp, strategy="temperature"
                )
            })

        # Top-k sampling
        for k in [5, 10]:
            results.append({
                "seed": seed,
                "strategy": "top_k_" + str(k),
                "temperature": None,
                "generated_text": generate_text(
                    model, seed, index_to_char, seq_length,
                    length=500, strategy="top_k", top_k=k
                )
            })

        # Top-p sampling
        results.append({
            "seed": seed,
            "strategy": "top_p_0.9",
            "temperature": None,
            "generated_text": generate_text(
                model, seed, index_to_char, seq_length,
                length=500, strategy="top_p", top_p=0.9
            )
        })

    table = pd.DataFrame(results)
    table.to_csv("./outputs/tables/generated_text_samples.csv", index=False)
    return table


def save_five_verses(model, index_to_char, seq_length=100):
    """Generate and save five sample verses using different seed phrases."""

    seeds = [
        "the world is ",
        "i remember ",
        "love is ",
        "the night was ",
        "when i was "
    ]

    poems = []

    for number, seed in enumerate(seeds, start=1):
        poem = generate_text(
            model, seed, index_to_char,
            seq_length=seq_length, length=500,
            temperature=0.9, strategy="top_k", top_k=10
        )
        poems.append({
            "verse": number,
            "seed": seed,
            "text": poem
        })

    poem_table = pd.DataFrame(poems)
    poem_table.to_csv("./outputs/tables/five_generated_verses.csv", index=False)

    with open("./outputs/tables/five_generated_verses.txt", "w", encoding="utf-8") as file:
        for poem in poems:
            file.write("VERSE " + str(poem["verse"]) + "\n")
            file.write(poem["text"] + "\n\n")

    return poem_table
