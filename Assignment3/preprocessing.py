import re
import html
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


# Download NLTK resources (only downloads missing resources)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


RAW_FILE = Path("data/raw/IMDB Dataset.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Keep words that can change the meaning or sentiment of a review.
SENTIMENT_WORDS = {
    "not", "no", "nor", "never", "neither", "very",
    "too", "but", "against", "don't", "didn't", "isn't",
    "wasn't", "couldn't", "wouldn't", "shouldn't"
}


def load_data(file_path = RAW_FILE):
    """Read the CSV in chunks so the full file is not loaded at once."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Download the Kaggle CSV and put it inside data/raw/."
        )

    chunks = pd.read_csv(
        file_path,
        usecols=["review", "sentiment"],
        chunksize=5000
    )

    data = pd.concat(chunks, ignore_index=True)
    data = data.dropna(subset=["review", "sentiment"])

    data["sentiment"] = (
        data["sentiment"].astype(str).str.lower().map(
            {"negative": 0, "positive": 1}
        )
    )
    data = data.dropna(subset=["sentiment"])
    data["sentiment"] = data["sentiment"].astype(int)

    return data


def check_class_distribution(data):
    """Show the number and percentage of positive and negative reviews."""
    counts = data["sentiment"].value_counts().sort_index()
    percentages = data["sentiment"].value_counts(
        normalize=True
    ).sort_index() * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages
    })

    distribution.index = ["Negative", "Positive"]
    print("\nClass distribution:")
    print(distribution)

    difference = abs(percentages.iloc[0] - percentages.iloc[1])
    if difference <= 5:
        print("\nThe dataset is approximately balanced.")
    else:
        print("\nThe dataset is not balanced.")

    distribution["count"].plot(kind="bar", title="Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "class_distribution.png")
    plt.show()


def clean_text(text):
    """Remove HTML, URLs and unwanted characters."""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text, keep_sentiment_words=True):
    """Clean, tokenize, remove stopwords and lemmatize one review."""
    text = clean_text(text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    stop_words = set(stopwords.words("english"))

    if keep_sentiment_words:
        stop_words = stop_words - SENTIMENT_WORDS

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
