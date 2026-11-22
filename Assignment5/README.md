# Assignment 5: Recurrent Neural Networks and Text Generation

This assignment applies recurrent neural networks to IMDB movie review sentiment classification and character-level text generation.

## Project Structure

- `Assignment5.ipynb` - Main notebook containing the full experiment workflow
- `sentiment_rnn.py` - Data preparation and Simple RNN, LSTM, BiLSTM, and GRU helpers
- `text_generation.py` - Character-level corpus preparation, LSTM training, and text generation helpers
- `sampling_strategies.py` - Greedy, temperature, top-k, and top-p sampling methods
- `data/` - Raw and processed datasets
- `models/` - Saved trained Keras models
- `outputs/` - Generated tables and figures

## Features

### Sentiment Classification

- Cleans and tokenizes IMDB movie reviews
- Splits data into training, validation, and test sets
- Compares Simple RNN, LSTM, Bidirectional LSTM, and GRU models
- Evaluates sequence length, model accuracy, loss, and training time
- Saves trained models, CSV summaries, and training plots

### Character-Level Text Generation

- Builds character sequences from the IMDB review corpus
- Trains a character-level LSTM
- Generates text using:
  - Greedy sampling
  - Temperature sampling
  - Top-k sampling
  - Top-p sampling
- Saves character mappings, loss tables, generated samples, and model checkpoints

## Requirements

Install the required Python packages:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter
```

## Usage

Place the IMDB dataset at:

```text
data/raw/IMDB Dataset.csv
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Assignment5.ipynb` and run all cells from top to bottom.

The notebook creates these directories automatically if they do not exist:

- `data/processed/`
- `models/`
- `outputs/tables/`
- `outputs/figures/`

## Main Outputs

- `models/simple_rnn.keras`
- `models/lstm.keras`
- `models/character_lstm.keras`
- `outputs/tables/sequence_length_summary.csv`
- `outputs/tables/text_generation_loss.csv`
- `outputs/figures/rnn_accuracy.png`
- `outputs/figures/rnn_loss.png`
- `outputs/figures/lstm_accuracy.png`
- `outputs/figures/lstm_loss.png`
- `outputs/figures/text_generation_loss.png`
