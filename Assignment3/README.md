# IMDB Sentiment Analysis

This project builds a sentiment classifier for IMDB movie reviews using classical NLP and machine learning techniques. The workflow includes text cleaning, feature extraction, vectorization experiments, feature selection, and model evaluation.

## Project structure

- `preprocessing.py` — cleans and preprocesses the raw reviews, downloads required NLTK resources, and creates the train/validation/test split.
- `vectorization.py` — compares Bag-of-Words and TF-IDF configurations and saves the resulting vectorizers.
- `feature_engineering.py` — creates custom review features and evaluates feature-selection methods.
- `models.py` — trains and evaluates several classifiers and saves the results.
- `Assignment3.ipynb` — interactive notebook walkthrough of the assignment.
- `data/raw/IMDB Dataset.csv` — raw Kaggle dataset (not included in the repo).
- `data/processed/` — cleaned datasets and experiment outputs.
- `models/` — trained models and saved vectorizers.
- `outputs/figures/` — model comparison plots and evaluation charts.

## Setup

1. Download the IMDB 50K dataset from Kaggle.
2. Place the CSV file at `data/raw/IMDB Dataset.csv`.
3. Create a Python environment and install dependencies:
   `pip install -r requirements.txt`

## Run the workflow

Run the project files in this order:

1. `python preprocessing.py`
2. `python vectorization.py`
3. `python feature_engineering.py`
4. `python models.py`
5. Open and run `Assignment3.ipynb` for the notebook-based walkthrough.

## Output folders

- Processed datasets: `data/processed/`
- Saved models: `models/`
- Evaluation plots: `outputs/figures/`

## Split strategy

The project uses a 60% training, 20% validation, and 20% test split for model training and evaluation.

## Models evaluated

The assignment compares multiple classifiers, including:

- Multinomial Naive Bayes
- Bernoulli Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- K-Nearest Neighbors
- Gradient Boosting
