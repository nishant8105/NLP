"""
Part 1 - Word2Vec training and analysis
Simple functions for Assignment 4.

Expected input:
- Text files (.txt), or
- CSV files with a column containing document text.

Example folders:
    data/raw/domain_corpus/
    data/processed/
    models/
    outputs/figures/
    outputs/tables/
"""

import re
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from scipy.stats import spearmanr


# ---------------------------------------------------------
# 1. Load documents
# ---------------------------------------------------------

def load_documents(data_path, text_column="text"):
    """
    Load documents from .txt files or CSV files.
    For CSV files, set text_column to the column containing text.
    Returns a list of document strings.
    """
    data_path = Path(data_path)
    documents = []

    if data_path.is_file():
        files = [data_path]
    else:
        files = list(data_path.rglob("*.txt")) + list(data_path.rglob("*.csv"))

    # Do not read the large pretrained GloVe file as a training document.
    files = [
        file_path for file_path in files
        if "glove" not in file_path.name.lower()
    ]

    for file_path in files:
        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append(text)

        elif file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
            if text_column not in df.columns:
                print(f"Skipping {file_path}: column '{text_column}' not found.")
                continue
            documents.extend(df[text_column].dropna().astype(str).tolist())

    print("Documents loaded:", len(documents))
    return documents


# ---------------------------------------------------------
# 2. Clean and tokenize documents
# ---------------------------------------------------------

def clean_text(text):
    """Lowercase text and remove special characters, keeping word hyphens."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_documents(documents):
    """
    Split each document into sentences, then words.
    Returns a list of tokenized sentences.
    """
    all_sentences = []

    for document in documents:
        document = clean_text(document)

        # A simple sentence split that does not need an extra language model.
        sentences = re.split(r"[.!?]+", document)

        for sentence in sentences:
            words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", sentence)
            if words:
                all_sentences.append(words)

    print("Sentences created:", len(all_sentences))
    return all_sentences


# ---------------------------------------------------------
# 3. Build vocabulary and word mappings
# ---------------------------------------------------------

def build_vocabulary(sentences, min_count=5):
    """Keep words appearing at least min_count times."""
    word_counts = Counter(word for sentence in sentences for word in sentence)

    vocabulary = {
        word: count for word, count in word_counts.items()
        if count >= min_count
    }

    # Index 0 is reserved for words not in the vocabulary.
    word_to_index = {
        word: index + 1 for index, word in enumerate(sorted(vocabulary))
    }
    index_to_word = {
        index: word for word, index in word_to_index.items()
    }

    print("Vocabulary size:", len(vocabulary))
    return vocabulary, word_to_index, index_to_word


def save_vocabulary(word_to_index, index_to_word, output_dir):
    """Save both vocabulary mappings as JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "word_to_index.json", "w", encoding="utf-8") as f:
        json.dump(word_to_index, f, indent=2)

    with open(output_dir / "index_to_word.json", "w", encoding="utf-8") as f:
        json.dump(index_to_word, f, indent=2)


# ---------------------------------------------------------
# 4. Prepare Skip-gram or CBOW training examples
# ---------------------------------------------------------

def generate_training_data(sentences, word_to_index, window=5, architecture="skipgram"):
    """
    Create simple training examples.

    Skip-gram:
        input = center word, target = context word

    CBOW:
        input = context words, target = center word

    Gensim creates these examples internally during model training.
    This function is included to help understand the training process.
    """
    inputs = []
    targets = []

    for sentence in sentences:
        ids = [word_to_index[w] for w in sentence if w in word_to_index]

        for center_position, center_word in enumerate(ids):
            start = max(0, center_position - window)
            end = min(len(ids), center_position + window + 1)

            context = [
                ids[i] for i in range(start, end)
                if i != center_position
            ]

            for context_word in context:
                if architecture.lower() == "skipgram":
                    inputs.append(center_word)
                    targets.append(context_word)

                elif architecture.lower() == "cbow":
                    inputs.append(context)
                    targets.append(center_word)

                else:
                    raise ValueError("architecture must be 'skipgram' or 'cbow'")

    return inputs, targets


# ---------------------------------------------------------
# 5. Train and save Word2Vec
# ---------------------------------------------------------

def train_word2vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10,
    negative=5,
    model_path="models/word2vec_custom.model"
):
    """Train a Word2Vec model and save it."""
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,                 # 1 = Skip-gram, 0 = CBOW
        epochs=epochs,
        negative=negative,
        seed=42
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    print("Model saved to:", model_path)
    return model


def run_hyperparameter_experiments(
    sentences,
    output_dir="models/word2vec_experiments",
    vector_sizes=(50, 100, 200, 300),
    windows=(2, 5, 10),
    architectures=("skipgram", "cbow"),
    epoch_values=(5, 10, 20)
):
    """
    Train models for the requested parameter combinations.
    This can take a long time because it trains many models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for vector_size in vector_sizes:
        for window in windows:
            for architecture in architectures:
                for epochs in epoch_values:
                    sg = 1 if architecture == "skipgram" else 0
                    model_name = (
                        f"w2v_{architecture}_dim{vector_size}_"
                        f"window{window}_epochs{epochs}.model"
                    )

                    print("Training:", model_name)
                    model = train_word2vec(
                        sentences=sentences,
                        vector_size=vector_size,
                        window=window,
                        sg=sg,
                        epochs=epochs,
                        model_path=output_dir / model_name
                    )

                    results.append({
                        "model": model_name,
                        "architecture": architecture,
                        "vector_size": vector_size,
                        "window": window,
                        "epochs": epochs,
                        "vocabulary_size": len(model.wv)
                    })

    results_df = pd.DataFrame(results)
    tables_dir = Path("outputs/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(tables_dir / "word2vec_experiments.csv", index=False)
    return results_df


# ---------------------------------------------------------
# 6. Similarity and analogy
# ---------------------------------------------------------

def find_similar_words(model, word, topn=10):
    """Return the most similar words. The word must be in the vocabulary."""
    if word not in model.wv:
        print(f"'{word}' is not in the model vocabulary.")
        return []
    return model.wv.most_similar(word, topn=topn)


def word_similarity(model, word1, word2):
    """Calculate cosine similarity between two words."""
    if word1 not in model.wv or word2 not in model.wv:
        print("One or both words are not in the model vocabulary.")
        return None
    return model.wv.similarity(word1, word2)


def word_analogy(model, positive, negative, topn=5):
    """Solve an analogy such as king - man + woman."""
    missing = [w for w in positive + negative if w not in model.wv]
    if missing:
        print("Words missing from vocabulary:", missing)
        return []
    return model.wv.most_similar(
        positive=positive, negative=negative, topn=topn
    )


# ---------------------------------------------------------
# 7. t-SNE visualization
# ---------------------------------------------------------

def plot_word_embeddings(
    model,
    output_path="outputs/figures/word2vec_tsne.png",
    top_n=500,
    categories=None
):
    """
    Plot the most frequent words in 2D using t-SNE.
    categories can be a dictionary, for example:
        {"doctor": "medical", "nurse": "medical", "court": "legal"}
    Words without a category are labelled 'other'.
    """
    words = model.wv.index_to_key[:top_n]
    if len(words) < 2:
        print("Not enough words to create a t-SNE plot.")
        return

    vectors = np.array([model.wv[word] for word in words])

    perplexity = min(30, len(words) - 1)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto"
    )
    points = tsne.fit_transform(vectors)

    categories = categories or {}
    labels = [categories.get(word, "other") for word in words]
    unique_labels = sorted(set(labels))

    plt.figure(figsize=(16, 12))
    for category in unique_labels:
        indexes = [i for i, label in enumerate(labels) if label == category]
        plt.scatter(
            points[indexes, 0],
            points[indexes, 1],
            label=category,
            alpha=0.7,
            s=18
        )

    # Label every point when plotting a small sample; otherwise labels overlap.
    if len(words) <= 100:
        for i, word in enumerate(words):
            plt.annotate(word, (points[i, 0], points[i, 1]), fontsize=7)

    plt.title("Word2Vec Embeddings - t-SNE")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()
    print("Figure saved to:", output_path)


# ---------------------------------------------------------
# 8. Intrinsic evaluation using word similarity data
# ---------------------------------------------------------

def evaluate_word_similarity(model, dataset_path, word1_col="word1",
                              word2_col="word2", score_col="score"):
    """
    Evaluate using a CSV containing word1, word2, and human similarity score.
    Returns Spearman correlation and the number of usable word pairs.
    """
    df = pd.read_csv(dataset_path)
    predicted = []
    actual = []

    for _, row in df.iterrows():
        word1 = str(row[word1_col]).lower()
        word2 = str(row[word2_col]).lower()

        if word1 in model.wv and word2 in model.wv:
            predicted.append(model.wv.similarity(word1, word2))
            actual.append(float(row[score_col]))

    if len(predicted) < 2:
        print("Not enough word pairs were found in the model vocabulary.")
        return None

    correlation, p_value = spearmanr(actual, predicted)

    result = {
        "spearman_correlation": float(correlation),
        "p_value": float(p_value),
        "usable_pairs": len(predicted),
        "total_pairs": len(df)
    }
    print(result)
    return result


def compare_with_pretrained(model, pretrained_model_path, word_pairs):
    """
    Compare custom and pretrained Gensim Word2Vec models on the same word pairs.
    pretrained_model_path must point to a Gensim .model file.
    """
    pretrained = Word2Vec.load(str(pretrained_model_path))
    rows = []

    for word1, word2 in word_pairs:
        row = {"word1": word1, "word2": word2}

        if word1 in model.wv and word2 in model.wv:
            row["custom_similarity"] = model.wv.similarity(word1, word2)
        else:
            row["custom_similarity"] = np.nan

        if word1 in pretrained.wv and word2 in pretrained.wv:
            row["pretrained_similarity"] = pretrained.wv.similarity(word1, word2)
        else:
            row["pretrained_similarity"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 9. Save embedding matrix
# ---------------------------------------------------------

def save_embedding_matrix(model, word_to_index, output_path="models/word2vec_embedding_matrix.npy"):
    """Create and save a matrix using the vocabulary mapping."""
    matrix = np.zeros((len(word_to_index) + 1, model.vector_size), dtype=np.float32)

    for word, index in word_to_index.items():
        if word in model.wv:
            matrix[index] = model.wv[word]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)

    print("Embedding matrix shape:", matrix.shape)
    print("Embedding matrix saved to:", output_path)
    return matrix
