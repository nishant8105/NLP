import re
import string
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag

  
def download_nltk_resources():
    for item in ["punkt", "punkt_tab", "stopwords", "wordnet",
                 "omw-1.4", "averaged_perceptron_tagger_eng"]:
        nltk.download(item, quiet=True)


porter = PorterStemmer()
snowball = SnowballStemmer("english")
lemma = WordNetLemmatizer()


def get_stopwords():
    words = set(stopwords.words("english"))
    words -= {"not", "no", "never"}
    return words | {"airline", "airlines", "flight", "flights"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+|(?<!\w)@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    contractions = {
        "can't": "can not", "won't": "will not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "isn't": "is not",
        "aren't": "are not", "wasn't": "was not", "weren't": "were not",
        "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
        "wouldn't": "would not", "couldn't": "could not",
        "shouldn't": "should not", "it's": "it is", "i'm": "i am",
        "you're": "you are", "we're": "we are", "they're": "they are",
        "i've": "i have", "i'll": "i will"
    }
    for short, full in contractions.items():
        text = re.sub(r"\b" + re.escape(short) + r"\b", full, text)

    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_tokenization(text):
    return word_tokenize(str(text))


def sentence_tokenization(text):
    return sent_tokenize(str(text))


def handle_edge_cases(text):
    text = str(text)
    return {
        "urls": re.findall(r"https?://\S+|www\.\S+", text),
        "mentions": re.findall(r"(?<!\w)@\w+", text),
        "hashtags": re.findall(r"(?<!\w)#\w+", text)
    }


def compare_tokenization(text):
    print("Simple split:", str(text).split())
    print("Word tokens:", word_tokenization(text))
    print("Sentences:", sentence_tokenization(text))
    print("URLs, mentions, hashtags:", handle_edge_cases(text))


def remove_stopwords(tokens):
    stop = get_stopwords()
    return [word for word in tokens if word.lower() not in stop]


def apply_porter_stemmer(tokens):
    return [porter.stem(word) for word in tokens]


def apply_snowball_stemmer(tokens):
    return [snowball.stem(word) for word in tokens]


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return "a"
    if tag.startswith("V"):
        return "v"
    if tag.startswith("R"):
        return "r"
    return "n"


def apply_wordnet_lemmatizer(tokens):
    return [lemma.lemmatize(word, get_wordnet_pos(tag))
            for word, tag in pos_tag(tokens)]


def compare_results(words):
    rows = []
    for word, tag in pos_tag(words):
        pos = get_wordnet_pos(tag)
        rows.append({
            "Original Word": word,
            "POS Tag": tag,
            "Porter Stem": porter.stem(word),
            "Snowball Stem": snowball.stem(word),
            "WordNet Lemma": lemma.lemmatize(word, pos)
        })
    return pd.DataFrame(rows)


def analyze_pos(text):
    tagged = pos_tag(word_tokenize(str(text)))
    return {
        "tagged_words": tagged,
        "nouns": [word for word, tag in tagged if tag.startswith("NN")],
        "adjectives": [word for word, tag in tagged if tag.startswith("JJ")],
        "pos_distribution": dict(Counter(tag for word, tag in tagged))
    }


def nltk_preprocess(text):
    words = remove_stopwords(word_tokenization(clean_text(text)))
    return " ".join(apply_wordnet_lemmatizer(words))


if __name__ == "__main__":
    download_nltk_resources()
    text = "I don't like this flight! The service was running badly. I am not happy. Visit https://example.com @john #Travel"

    print("1. TOKENIZATION")
    compare_tokenization(text)

    cleaned = clean_text(text)
    print("\n2. CLEANED TEXT:", cleaned)
    print("\n3. STOPWORD REMOVAL:", remove_stopwords(word_tokenization(cleaned)))

    words = ["running", "runs", "ran", "better", "good", "best"]
    print("\n4. PORTER STEMMING:", apply_porter_stemmer(words))
    print("\n5. SNOWBALL STEMMING:", apply_snowball_stemmer(words))
    print("\n6. LEMMATIZATION:", apply_wordnet_lemmatizer(words))
    print("\n7. STEMMING COMPARISON")
    print(compare_results(words).to_string(index=False))
    print("\n8. POS ANALYSIS:", analyze_pos(text))
    print("\n9. COMPLETE PIPELINE:", nltk_preprocess(text))
