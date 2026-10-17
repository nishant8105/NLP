import json
import time
from collections import Counter

import pandas as pd
import spacy
from spacy import displacy

  
def load_spacy_model(name="en_core_web_sm"):
    try:
        return spacy.load(name)
    except OSError:
        raise RuntimeError(f"Install the model with: python -m spacy download {name}")


nlp = load_spacy_model()
NEGATION_WORDS = {"not", "no", "never"}


def process_text_spacy(text, nlp=nlp):
    doc = nlp(str(text))
    return pd.DataFrame([{
        "Token": t.text, "Lemma": t.lemma_, "POS": t.pos_,
        "Tag": t.tag_, "Is_Stopword": t.is_stop, "Is_Punctuation": t.is_punct
    } for t in doc])


def lemmatize_text(text, nlp=nlp):
    return [t.lemma_ for t in nlp(str(text))]


def pos_tagging_spacy(text, nlp=nlp):
    return [(t.text, t.pos_) for t in nlp(str(text))]


def dependency_parsing(text, nlp=nlp, visualize=False):
    doc = nlp(str(text))
    data = [{"Token": t.text, "Head": t.head.text,
             "Dependency": t.dep_, "POS": t.pos_} for t in doc]
    if visualize:
        displacy.render(doc, style="dep", jupyter=True)
    return pd.DataFrame(data)


def extract_entities(text, nlp=nlp):
    return pd.DataFrame([{
        "Entity": e.text, "Label": e.label_,
        "Description": spacy.explain(e.label_)
    } for e in nlp(str(text)).ents],
        columns=["Entity", "Label", "Description"])


def count_entity_types(text, nlp=nlp):
    counts = Counter(e.label_ for e in nlp(str(text)).ents)
    return pd.DataFrame(counts.items(), columns=["Entity Type", "Count"]).sort_values(
        "Count", ascending=False
    ).reset_index(drop=True)


def entity_frequency(texts, nlp=nlp):
    counts = Counter()
    for doc in nlp.pipe((str(t) for t in texts), batch_size=128):
        counts.update(e.label_ for e in doc.ents)
    return pd.DataFrame(counts.items(), columns=["Entity Type", "Count"]).sort_values(
        "Count", ascending=False
    ).reset_index(drop=True)


def plot_entity_frequency(texts, nlp=nlp):
    import matplotlib.pyplot as plt
    counts = entity_frequency(texts, nlp)
    if counts.empty:
        print("No named entities found.")
        return counts
    counts.sort_values("Count").plot.barh(x="Entity Type", y="Count", legend=False)
    plt.title("Named Entity Frequency")
    plt.tight_layout()
    plt.show()
    return counts


def add_custom_stopwords(words, nlp=nlp):
    for word in words:
        word = str(word).lower().strip()
        if word and word not in NEGATION_WORDS:
            nlp.vocab[word].is_stop = True


def create_custom_entity_ruler(nlp=nlp):
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    ruler.add_patterns([
        {"label": "AIRLINE", "pattern": "Air India"},
        {"label": "AIRLINE", "pattern": "IndiGo"},
        {"label": "AIRPORT", "pattern": "Chhatrapati Shivaji Maharaj International Airport"},
        {"label": "AIRPORT_CODE", "pattern": "BOM"},
        {"label": "BOOKING_REF", "pattern": [{"TEXT": {"REGEX": "^[A-Z]{6}$"}}]}
    ])
    return nlp


def custom_filtering(text, nlp=nlp, allowed_pos=None, remove_stopwords=True):
    result = []
    for token in nlp(str(text)):
        if token.is_punct or token.is_space:
            continue
        if remove_stopwords and token.is_stop and token.lower_ not in NEGATION_WORDS:
            continue
        if allowed_pos is None or token.pos_ in allowed_pos:
            result.append(token.text)
    return result


def spacy_preprocess(text, nlp=nlp, allowed_pos=None, remove_stopwords=True):
    tokens = []
    for token in nlp(str(text)):
        if token.is_punct or token.is_space:
            continue
        if remove_stopwords and token.is_stop and token.lower_ not in NEGATION_WORDS:
            continue
        if allowed_pos is not None and token.pos_ not in allowed_pos:
            continue
        if token.lemma_.strip():
            tokens.append(token.lemma_.lower().strip())
    return " ".join(tokens)


def spacy_preprocess_batch(texts, nlp=nlp, batch_size=128,
                           allowed_pos=None, remove_stopwords=True):
    return [
        spacy_preprocess_doc(doc, allowed_pos, remove_stopwords)
        for doc in nlp.pipe((str(t) for t in texts), batch_size=batch_size)
    ]


def spacy_preprocess_doc(doc, allowed_pos=None, remove_stopwords=True):
    words = []
    for token in doc:
        if token.is_punct or token.is_space:
            continue
        if remove_stopwords and token.is_stop and token.lower_ not in NEGATION_WORDS:
            continue
        if allowed_pos is not None and token.pos_ not in allowed_pos:
            continue
        if token.lemma_.strip():
            words.append(token.lemma_.lower().strip())
    return " ".join(words)


def export_entities_json(texts, output_path="entities.json", nlp=nlp):
    records = []
    for i, doc in enumerate(nlp.pipe((str(t) for t in texts), batch_size=128)):
        for ent in doc.ents:
            records.append({
                "tweet_index": i, "entity": ent.text, "label": ent.label_,
                "description": spacy.explain(ent.label_)
            })
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)
    return records


def benchmark_spacy(texts, nlp=nlp, batch_size=128):
    texts = list(texts)
    start = time.perf_counter()
    output = spacy_preprocess_batch(texts, nlp, batch_size)
    seconds = time.perf_counter() - start
    return {
        "library": "spaCy", "tweets": len(texts), "elapsed_seconds": seconds,
        "tweets_per_second": len(texts) / seconds if seconds else 0,
        "outputs": output
    }


if __name__ == "__main__":
    text = ("Air India provided excellent services to Mumbai passengers on Monday. "
            "Tim Cook visited London. The airline delayed several flights, "
            "and I am not happy.")

    print("1. TOKENS, LEMMAS AND POS")
    print(process_text_spacy(text).to_string(index=False))
    print("\n2. LEMMATIZATION:", lemmatize_text(text))
    print("\n3. POS TAGGING:", pos_tagging_spacy(text))
    print("\n4. DEPENDENCIES")
    print(dependency_parsing(text).to_string(index=False))
    print("\n5. NAMED ENTITIES")
    print(extract_entities(text).to_string(index=False))

    add_custom_stopwords(["provided", "services", "airline"])
    print("\n6. CUSTOM FILTERING:", custom_filtering(text))
    create_custom_entity_ruler()
    print("\n7. CUSTOM ENTITIES")
    print(extract_entities("Air India operates from BOM. My booking reference is ABCDEF."))
    print("\n8. COMPLETE PREPROCESSING:", spacy_preprocess(text))
    print("\n9. NOUNS:", custom_filtering(text, allowed_pos={"NOUN", "PROPN"}))
    print("\n10. VERBS:", custom_filtering(text, allowed_pos={"VERB"}))
