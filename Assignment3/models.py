# Machine Learning Models for IMDB Sentiment Analysis

import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_auc_score
)


DATA = Path("data/processed")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs/figures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load the data saved in Part 1
def load_data():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")

    X_train = train["review"].fillna("").astype(str)
    y_train = train["sentiment"]
    X_test = test["review"].fillna("").astype(str)
    y_test = test["sentiment"]

    return X_train, X_test, y_train, y_test


# Create pipelines for all six models
def create_models():
    models = {
        "Multinomial NB": Pipeline([
            ("vectorizer", CountVectorizer(max_features=5000)),
            ("model", MultinomialNB())
        ]),

        "Bernoulli NB": Pipeline([
            ("vectorizer", CountVectorizer(max_features=5000, binary=True)),
            ("model", BernoulliNB())
        ]),

        "Logistic Regression": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=5000)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),

        "Linear SVM": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=5000)),
            ("model", LinearSVC(max_iter=3000))
        ]),

        "Random Forest": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=3000)),
            ("model", RandomForestClassifier(random_state=42, n_jobs=-1))
        ]),

        "KNN": Pipeline([
            ("vectorizer", TfidfVectorizer(max_features=3000)),
            ("model", KNeighborsClassifier())
        ])
    }

    models["Gradient Boosting"] = Pipeline([
        ("vectorizer", TfidfVectorizer(max_features=3000)),
        ("reduce", TruncatedSVD(n_components=100, random_state=42)),
        ("model", GradientBoostingClassifier(random_state=42))
    ])

    return models


# Parameter settings for the three models we tune
def get_parameter_grids():
    return {
        "Multinomial NB": {
            "model__alpha": [0.5, 1.0, 2.0]
        },

        "Logistic Regression": {
            "model__C": [0.1, 1, 10],
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear"]
        },

        "Linear SVM": {
            "model__C": [0.1, 1, 10],
            "model__max_iter": [2000, 5000]
        }
    }


# Train all models and tune three models with 5-fold cross-validation
def train_models(X_train, y_train):
    models = create_models()
    parameter_grids = get_parameter_grids()

    trained_models = {}
    training_times = {}
    best_parameters = {}

    for name, pipeline in models.items():
        print("\n Training:", name)
        start = time.time()

        if name in parameter_grids:
            search = GridSearchCV(
                pipeline,
                parameter_grids[name],
                cv=5,
                scoring="f1_macro",
                n_jobs=1
            )
            search.fit(X_train, y_train)

            trained_models[name] = search.best_estimator_
            best_parameters[name] = search.best_params_

            print("Best parameters:", search.best_params_)
            print("Best CV score:", search.best_score_)

        else:
            pipeline.fit(X_train, y_train)
            trained_models[name] = pipeline

        training_times[name] = time.time() - start
        print("Training time:", round(training_times[name], 2), "seconds")

    return trained_models, training_times, best_parameters


# Get scores needed for ROC-AUC
def get_model_scores(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    return None


# Evaluate one model
def evaluate_model(name, model, X_test, y_test):
    start = time.time()
    predictions = model.predict(X_test)
    prediction_time = time.time() - start

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_test, predictions, average="macro", zero_division=0
        )
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_test, predictions, average="weighted", zero_division=0
        )
    )

    scores = get_model_scores(model, X_test)

    if scores is not None:
        auc = roc_auc_score(y_test, scores)
    else:
        auc = float("nan")

    print("\nClassification Report:", name)
    print(classification_report(
        y_test, predictions,
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    # Draw confusion matrix
    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(
        matrix, display_labels=["Negative", "Positive"]
    )
    display.plot(cmap="Blues")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / (name.replace(" ", "_") + "_confusion.png"))
    plt.close()

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc": auc,
        "prediction_time": prediction_time
    }


# Evaluate all trained models
def evaluate_models(models, X_test, y_test):
    results = []

    for name, model in models.items():
        print("\nEvaluating:", name)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)

    results = pd.DataFrame(results)
    results.to_csv(DATA / "model_results.csv", index=False)

    # Compare accuracy and F1 scores
    results.set_index("model")[["accuracy", "f1_macro", "f1_weighted"]].plot(
        kind="bar", figsize=(10, 5)
    )
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png")
    plt.show()

    return results


# Save trained models and tuning results
def save_models(models, training_times, best_parameters):
    joblib.dump(models, MODEL_DIR / "trained_models.pkl")

    pd.DataFrame({
        "model": list(training_times.keys()),
        "training_time": list(training_times.values())
    }).to_csv(DATA / "training_times.csv", index=False)

    pd.DataFrame([
        {"model": name, "best_parameters": str(params)}
        for name, params in best_parameters.items()
    ]).to_csv(DATA / "best_parameters.csv", index=False)



