import os
import time
import tracemalloc
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import nltk_pipeline as nltk
import spacy_pipeline as spacy

DATA = "Tweets.csv"
OUT = "benchmark_outputs"
SIZES = [1000, 5000, 10000]
TEXT = "text"
os.makedirs(OUT, exist_ok=True)

  
def run_nltk(texts):
    return [nltk.nltk_preprocess(t) for t in texts]


def run_spacy(texts):
    texts = [nltk.clean_text(t) for t in texts]
    return spacy.spacy_preprocess_batch(texts, nlp=spacy.nlp, batch_size=128)


def measure(name, function, texts):
    tracemalloc.start()
    start = time.time()
    output = function(texts)
    seconds = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    words = [t.split() if isinstance(t, str) else [] for t in output]
    all_words = [w for row in words for w in row]

    return output, {
        "library": name,
        "dataset_size": len(texts),
        "seconds": round(seconds, 3),
        "tweets_per_second": round(len(texts) / seconds, 2) if seconds else 0,
        "average_tokens": round(sum(map(len, words)) / len(words), 2) if words else 0,
        "total_tokens": len(all_words),
        "unique_words": len(set(all_words)),
        "peak_memory_mb": round(peak / (1024 ** 2), 2),
    }


def main():
    nltk.download_nltk_resources()
    df = pd.read_csv(DATA).dropna(subset=[TEXT]).reset_index(drop=True)
    df[TEXT] = df[TEXT].astype(str)
    print("Tweets loaded:", len(df))

    spacy.create_custom_entity_ruler(spacy.nlp)
    sample = df.sample(n=min(max(SIZES), len(df)), random_state=42)
    results, cache = [], {}

    for size in SIZES:
        if size > len(sample):
            continue
        texts = sample[TEXT].iloc[:size].tolist()
        print(f"\nTesting {size} tweets")

        for name, function in [("NLTK", run_nltk), ("spaCy", run_spacy)]:
            output, stats = measure(name, function, texts)
            results.append(stats)
            cache[(name, size)] = (texts, output)
            print(name, stats)

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{OUT}/benchmark_results.csv", index=False)
    print("\nBenchmark summary:")
    print(result_df.to_string(index=False))

    largest = max(size for _, size in cache)
    for name in ["NLTK", "spaCy"]:
        texts, output = cache[(name, largest)]
        pd.DataFrame({"original_text": texts, "processed_text": output}).to_csv(
            f"{OUT}/{name.lower()}_processed.csv", index=False
        )

    # POS tags and named entities from spaCy
    pos, entities = Counter(), Counter()
    texts = cache[("spaCy", largest)][0]
    for doc in spacy.nlp.pipe(texts, batch_size=128):
        pos.update(token.pos_ for token in doc)
        entities.update(ent.label_ for ent in doc.ents)

    pd.DataFrame(pos.items(), columns=["POS", "Count"]).sort_values(
        "Count", ascending=False
    ).to_csv(f"{OUT}/spacy_pos_distribution.csv", index=False)
    pd.DataFrame(entities.items(), columns=["Entity Type", "Count"]).sort_values(
        "Count", ascending=False
    ).to_csv(f"{OUT}/spacy_entity_counts.csv", index=False)

    # Small sample for manual quality checking
    quality = sample.sample(n=min(100, len(sample)), random_state=42).copy()
    quality["nltk_processed"] = run_nltk(quality[TEXT].tolist())
    quality["spacy_processed"] = run_spacy(quality[TEXT].tolist())
    quality["nltk_lemma_quality"] = ""
    quality["spacy_lemma_quality"] = ""
    quality["notes"] = ""
    quality.to_csv(f"{OUT}/lemmatization_quality_review.csv", index=False)

    # Comparison graphs
    for column, title, filename in [
        ("seconds", "Processing Time", "processing_time.png"),
        ("tweets_per_second", "Processing Speed", "throughput.png"),
        ("average_tokens", "Average Tokens per Tweet", "average_tokens.png"),
    ]:
        for name in result_df["library"].unique():
            part = result_df[result_df["library"] == name]
            plt.plot(part["dataset_size"], part[column], marker="o", label=name)
        plt.xlabel("Number of tweets")
        plt.ylabel(title)
        plt.title(title + ": NLTK vs spaCy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT}/{filename}")
        plt.close()

    print(f"\nDone! Check the '{OUT}' folder for results.")


if __name__ == "__main__":
    main()
