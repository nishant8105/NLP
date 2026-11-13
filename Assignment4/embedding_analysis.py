from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


# ---------------------------------------------------------
# Common helper
# ---------------------------------------------------------

def make_parent_folder(file_path):
    """Create the folder where a file will be saved."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# PART 4 - EMBEDDING VISUALIZATION AND ANALYSIS
# =========================================================

# ---------------------------------------------------------
# 4.1 Extract emotion word vectors
# ---------------------------------------------------------

def get_emotion_word_vectors(model, embedding_type="word2vec"):
    """
    Get vectors for happy, sad and anger words.

    For Word2Vec:
        model is a trained Gensim Word2Vec model.

    For GloVe:
        model is the dictionary returned by load_glove().
    """
    emotion_words = {
        "Happy": ["joy", "happy", "excited", "thrilled"],
        "Sad": ["sad", "depressed", "unhappy", "miserable"],
        "Anger": ["angry", "furious", "mad", "rage"]
    }

    vectors = []
    words = []
    categories = []

    for category, word_list in emotion_words.items():
        for word in word_list:
            if embedding_type.lower() == "word2vec":
                if word in model.wv:
                    vector = model.wv[word]
                else:
                    print("Word not found:", word)
                    continue
            else:
                if word in model:
                    vector = model[word]
                else:
                    print("Word not found:", word)
                    continue

            words.append(word)
            vectors.append(vector)
            categories.append(category)

    if not vectors:
        raise ValueError("None of the requested emotion words were found.")

    return words, np.array(vectors), categories


# ---------------------------------------------------------
# 4.1 t-SNE in 2D and 3D
# ---------------------------------------------------------

def plot_emotion_words_tsne(words, vectors, categories,
                            dimensions=2,
                            output_path=None):
    """Plot emotion word vectors in 2D or 3D using t-SNE."""
    if dimensions not in [2, 3]:
        raise ValueError("dimensions must be 2 or 3")

    if len(vectors) <= dimensions:
        raise ValueError("Not enough words for this t-SNE dimension.")

    perplexity = min(5, len(vectors) - 1)

    tsne = TSNE(
        n_components=dimensions,
        random_state=42,
        perplexity=perplexity,
        init="random",
        learning_rate="auto"
    )

    points = tsne.fit_transform(vectors)
    unique_categories = sorted(set(categories))

    if dimensions == 2:
        plt.figure(figsize=(9, 7))

        for category in unique_categories:
            indexes = [
                i for i, value in enumerate(categories)
                if value == category
            ]
            plt.scatter(
                points[indexes, 0],
                points[indexes, 1],
                label=category,
                s=80
            )

        for i, word in enumerate(words):
            plt.annotate(word, (points[i, 0], points[i, 1]))

        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")

    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        for category in unique_categories:
            indexes = [
                i for i, value in enumerate(categories)
                if value == category
            ]
            ax.scatter(
                points[indexes, 0],
                points[indexes, 1],
                points[indexes, 2],
                label=category,
                s=80
            )

        for i, word in enumerate(words):
            ax.text(points[i, 0], points[i, 1], points[i, 2], word)

        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_zlabel("t-SNE 3")

    plt.title(f"Emotion Word Embeddings - t-SNE ({dimensions}D)")
    plt.legend()
    plt.tight_layout()

    if output_path:
        make_parent_folder(output_path)
        plt.savefig(output_path, dpi=300)

    plt.show()
    plt.close()

    return points


# ---------------------------------------------------------
# 4.2 Embedding arithmetic and semantic relationships
# ---------------------------------------------------------

def embedding_analogy(model, positive, negative, topn=5):
    """Run a Word2Vec analogy such as happy - joy + sadness."""
    missing = [
        word for word in positive + negative
        if word not in model.wv
    ]

    if missing:
        print("Words not found in Word2Vec:", missing)
        return []

    return model.wv.most_similar(
        positive=positive,
        negative=negative,
        topn=topn
    )


def compare_word_similarity(model, word_pairs, embedding_type="word2vec"):
    """
    Calculate cosine similarity for word pairs.
    For Word2Vec, pass a Gensim model.
    For GloVe, pass the GloVe dictionary.
    """
    results = []

    for word1, word2 in word_pairs:
        try:
            if embedding_type.lower() == "word2vec":
                if word1 not in model.wv or word2 not in model.wv:
                    score = np.nan
                else:
                    score = model.wv.similarity(word1, word2)
            else:
                if word1 not in model or word2 not in model:
                    score = np.nan
                else:
                    v1 = model[word1]
                    v2 = model[word2]
                    score = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
                    )

            results.append({
                "word1": word1,
                "word2": word2,
                "cosine_similarity": score
            })

        except Exception as error:
            print("Could not compare", word1, word2, ":", error)

    return pd.DataFrame(results)


def get_intensity_and_opposition(model, embedding_type="word2vec"):
    """Compare emotion intensity and emotion opposition word pairs."""
    pairs = [
        ("happy", "ecstatic"),
        ("joy", "sadness"),
        ("love", "hate"),
        ("happy", "thrilled")
    ]

    return compare_word_similarity(
        model,
        pairs,
        embedding_type=embedding_type
    )


# ---------------------------------------------------------
# 4.3 Optional attention model and visualization
# ---------------------------------------------------------

def build_attention_model(embedding_matrix, max_length, num_classes):
    """
    Build a small model with attention.
    Returns:
        model: predicts emotion probabilities
        attention_model: returns attention weights for each token

    This is an optional model; the Dense/CNN models from Part 2 do not
    contain attention layers.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input, Embedding, Dense, Dropout, Layer
    )
    from tensorflow.keras.models import Model

    class SimpleAttention(Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            self.attention_weights_layer = Dense(1, activation="tanh")
            super().build(input_shape)

        def call(self, inputs):
            scores = self.attention_weights_layer(inputs)
            weights = tf.nn.softmax(scores, axis=1)
            context = tf.reduce_sum(inputs * weights, axis=1)
            return context, tf.squeeze(weights, axis=-1)

    text_input = Input(shape=(max_length,))
    embedding = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=True
    )(text_input)

    context, attention_weights = SimpleAttention()(embedding)
    hidden = Dense(128, activation="relu")(context)
    hidden = Dropout(0.4)(hidden)
    output = Dense(num_classes, activation="sigmoid")(hidden)

    model = Model(text_input, output)
    attention_model = Model(text_input, [output, attention_weights])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model, attention_model


def plot_attention_for_text(text, tokenizer, attention_model,
                            max_length=100,
                            output_path="outputs/figures/attention_heatmap.png"):
    """
    Display a heatmap showing attention weights for each input word.
    Use attention_model returned by build_attention_model().
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    words = str(text).lower().split()
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    probabilities, weights = attention_model.predict(padded, verbose=0)
    weights = weights[0]

    shown_words = words[:max_length]
    shown_weights = weights[:len(shown_words)]

    if len(shown_words) == 0:
        print("No words to display.")
        return

    plt.figure(figsize=(max(8, len(shown_words) * 0.5), 3))
    plt.imshow(
        [shown_weights],
        cmap="YlOrRd",
        aspect="auto"
    )
    plt.xticks(
        range(len(shown_words)),
        shown_words,
        rotation=45,
        ha="right"
    )
    plt.yticks([])
    plt.colorbar(label="Attention weight")
    plt.title("Attention Weights for Input Text")
    plt.tight_layout()

    make_parent_folder(output_path)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    return probabilities, shown_weights


# ---------------------------------------------------------
# 4.4 Emotion centroids and distances
# ---------------------------------------------------------

def create_comment_vectors(x_padded, embedding_matrix):
    """
    Convert each padded comment into one vector by averaging its
    non-padding word embeddings.
    """
    comment_vectors = []

    for sequence in x_padded:
        word_vectors = embedding_matrix[sequence]
        valid_vectors = word_vectors[sequence != 0]

        if len(valid_vectors) == 0:
            comment_vectors.append(
                np.zeros(embedding_matrix.shape[1])
            )
        else:
            comment_vectors.append(np.mean(valid_vectors, axis=0))

    return np.array(comment_vectors)


def calculate_emotion_centroids(comment_vectors, labels, emotion_names):
    """Calculate the average comment vector for each emotion."""
    centroids = []

    for emotion_index, emotion in enumerate(emotion_names):
        indexes = np.where(labels[:, emotion_index] == 1)[0]

        if len(indexes) == 0:
            centroid = np.zeros(comment_vectors.shape[1])
        else:
            centroid = np.mean(comment_vectors[indexes], axis=0)

        centroids.append(centroid)

    return np.array(centroids)


def calculate_centroid_distances(centroids, emotion_names):
    """Calculate Euclidean distances between emotion centroids."""
    distances = cdist(centroids, centroids, metric="euclidean")

    return pd.DataFrame(
        distances,
        index=emotion_names,
        columns=emotion_names
    )


def calculate_centroid_similarity(centroids, emotion_names):
    """Calculate cosine similarity between emotion centroids."""
    similarities = cosine_similarity(centroids)

    return pd.DataFrame(
        similarities,
        index=emotion_names,
        columns=emotion_names
    )


def plot_emotion_similarity_matrix(similarity_df,
                                   output_path="outputs/figures/emotion_similarity_matrix.png"):
    """Plot the emotion centroid similarity matrix."""
    plt.figure(figsize=(12, 10))
    plt.imshow(similarity_df.values, cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Cosine similarity")
    plt.xticks(
        range(len(similarity_df.columns)),
        similarity_df.columns,
        rotation=90
    )
    plt.yticks(
        range(len(similarity_df.index)),
        similarity_df.index
    )
    plt.title("Emotion Centroid Similarity Matrix")
    plt.tight_layout()

    make_parent_folder(output_path)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()


def plot_centroids_tsne(centroids, emotion_names, dimensions=2,
                        output_path="outputs/figures/emotion_centroids_tsne.png"):
    """Plot emotion centroids using 2D or 3D t-SNE."""
    if dimensions not in [2, 3]:
        raise ValueError("dimensions must be 2 or 3")

    perplexity = min(5, len(centroids) - 1)
    tsne = TSNE(
        n_components=dimensions,
        random_state=42,
        perplexity=perplexity,
        init="random",
        learning_rate="auto"
    )
    points = tsne.fit_transform(centroids)

    if dimensions == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(points[:, 0], points[:, 1], s=70)

        for i, name in enumerate(emotion_names):
            plt.annotate(name, (points[i, 0], points[i, 1]))

    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=70)

        for i, name in enumerate(emotion_names):
            ax.text(points[i, 0], points[i, 1], points[i, 2], name)

    plt.title(f"Emotion Centroids - t-SNE ({dimensions}D)")
    plt.tight_layout()

    make_parent_folder(output_path)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    return points


# =========================================================
# PART 5 - ERROR ANALYSIS
# =========================================================

# ---------------------------------------------------------
# 5.1 Confusion matrix
# ---------------------------------------------------------

def make_single_label_predictions(y_true, probabilities):
    """
    Convert multi-label arrays to one dominant emotion per comment.
    Used only to create the requested 27x27 confusion matrix.
    """
    true_class = np.argmax(y_true, axis=1)
    predicted_class = np.argmax(probabilities, axis=1)
    return true_class, predicted_class


def plot_confusion_matrix_27(y_true, probabilities, emotion_names,
                              output_path="outputs/figures/confusion_matrix_27x27.png"):
    """
    Plot a 27x27 dominant-emotion confusion matrix.
    For datasets with a different number of labels, matrix size follows
    the actual number of emotion names.
    """
    true_class, predicted_class = make_single_label_predictions(
        y_true, probabilities
    )

    matrix = confusion_matrix(
        true_class,
        predicted_class,
        labels=np.arange(len(emotion_names))
    )

    fig, ax = plt.subplots(figsize=(14, 12))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=emotion_names
    )
    display.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=True,
        values_format="d"
    )
    plt.title("Confusion Matrix - Dominant Emotion")
    plt.tight_layout()

    make_parent_folder(output_path)
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    return matrix


# ---------------------------------------------------------
# 5.2 Find commonly confused emotion pairs
# ---------------------------------------------------------

def find_confused_emotion_pairs(matrix, emotion_names, top_n=10):
    """Find emotion pairs with the largest off-diagonal confusion counts."""
    pairs = []

    for true_index in range(matrix.shape[0]):
        for predicted_index in range(matrix.shape[1]):
            if true_index != predicted_index:
                count = matrix[true_index, predicted_index]

                if count > 0:
                    pairs.append({
                        "true_emotion": emotion_names[true_index],
                        "predicted_emotion": emotion_names[predicted_index],
                        "count": int(count)
                    })

    result = pd.DataFrame(pairs)

    if len(result) == 0:
        return result

    return result.sort_values(
        "count", ascending=False
    ).head(top_n)


# ---------------------------------------------------------
# 5.3 Show misclassified examples
# ---------------------------------------------------------

def get_misclassified_examples(
    texts,
    y_true,
    probabilities,
    emotion_names,
    true_emotion,
    predicted_emotion,
    number=10
):
    """Get examples for one true-emotion / predicted-emotion pair."""
    true_class, predicted_class = make_single_label_predictions(
        y_true, probabilities
    )

    true_index = emotion_names.index(true_emotion)
    predicted_index = emotion_names.index(predicted_emotion)

    indexes = np.where(
        (true_class == true_index) &
        (predicted_class == predicted_index)
    )[0][:number]

    examples = []
    for index in indexes:
        examples.append({
            "text": texts[index],
            "true_emotion": true_emotion,
            "predicted_emotion": predicted_emotion,
            "predicted_confidence": float(
                probabilities[index, predicted_index]
            )
        })

    return pd.DataFrame(examples)


def save_error_examples(examples, output_path="outputs/tables/error_examples.csv"):
    """Save misclassified examples to CSV."""
    make_parent_folder(output_path)
    examples.to_csv(output_path, index=False)


# ---------------------------------------------------------
# 5.4 Performance by comment length
# ---------------------------------------------------------

def analyze_performance_by_length(texts, y_true, probabilities,
                                  output_path="outputs/tables/performance_by_length.csv"):
    """Group comments by length and calculate dominant-label accuracy."""
    true_class, predicted_class = make_single_label_predictions(
        y_true, probabilities
    )

    rows = []
    for i, text in enumerate(texts):
        length = len(str(text).split())

        if length <= 10:
            group = "Short (1-10 words)"
        elif length <= 30:
            group = "Medium (11-30 words)"
        else:
            group = "Long (more than 30 words)"

        rows.append({
            "length_group": group,
            "correct": int(true_class[i] == predicted_class[i])
        })

    df = pd.DataFrame(rows)
    result = df.groupby("length_group").agg(
        number_of_comments=("correct", "count"),
        accuracy=("correct", "mean")
    ).reset_index()

    make_parent_folder(output_path)
    result.to_csv(output_path, index=False)
    return result


# ---------------------------------------------------------
# 5.5 Performance by emotion frequency
# ---------------------------------------------------------

def analyze_performance_by_frequency(y_true, probabilities, emotion_names,
                                     output_path="outputs/tables/performance_by_emotion_frequency.csv"):
    """Calculate per-emotion F1 and group emotions by frequency."""
    from sklearn.metrics import f1_score

    predictions = (probabilities >= 0.5).astype(int)
    counts = y_true.sum(axis=0)
    scores = f1_score(
        y_true,
        predictions,
        average=None,
        zero_division=0
    )

    result = pd.DataFrame({
        "emotion": emotion_names,
        "number_of_examples": counts.astype(int),
        "f1_score": scores
    })

    median_count = np.median(counts)
    result["frequency_group"] = np.where(
        counts < median_count, "Rare",
        "Common"
    )

    make_parent_folder(output_path)
    result.to_csv(output_path, index=False)
    return result


# ---------------------------------------------------------
# 5.6 Lexical complexity analysis
# ---------------------------------------------------------

def analyze_lexical_complexity(texts, y_true, probabilities,
                               output_path="outputs/tables/performance_by_lexical_complexity.csv"):
    """
    Use type-token ratio (unique words / total words) as a simple
    lexical diversity measure. This is only a rough complexity proxy.
    """
    true_class, predicted_class = make_single_label_predictions(
        y_true, probabilities
    )

    rows = []
    for i, text in enumerate(texts):
        words = str(text).lower().split()
        total_words = len(words)
        unique_words = len(set(words))
        diversity = unique_words / total_words if total_words else 0

        rows.append({
            "lexical_diversity": diversity,
            "correct": int(true_class[i] == predicted_class[i])
        })

    df = pd.DataFrame(rows)
    df["complexity_group"] = pd.cut(
        df["lexical_diversity"],
        bins=[-0.001, 0.4, 0.7, 1.001],
        labels=["Low", "Medium", "High"]
    )

    result = df.groupby(
        "complexity_group",
        observed=False
    ).agg(
        number_of_comments=("correct", "count"),
        accuracy=("correct", "mean")
    ).reset_index()

    make_parent_folder(output_path)
    result.to_csv(output_path, index=False)
    return result
