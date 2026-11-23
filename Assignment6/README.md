# Assignment 6: Transformer-Based Sentiment Analysis

This assignment uses the SST-2 sentiment dataset to prepare text, fine-tune and compare transformer models, visualize attention, explore advanced training techniques, and deploy a sentiment classifier with Streamlit.

## Project files

- `assignment6.ipynb` - End-to-end notebook for data preparation, training, evaluation, visualization, and deployment.
- `app.py` - Streamlit application for predicting the sentiment of a user-provided sentence.
- `data/processed/` - Tokenized SST-2 train, validation, and test datasets created by the notebook.
- `models/` - Tokenizers and fine-tuned model checkpoints created or used by the notebook.
- `outputs/figures/` and `outputs/tables/` - Generated visualizations and evaluation summaries.
- `results/` - Trainer checkpoints created during training.

## Requirements

Use Python 3.9 or later. Install the main dependencies:

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib jupyter streamlit
```

Some optional notebook sections require extra packages, including `nltk`, `sentencepiece`, `onnx`, and `onnxruntime`.

## Run the notebook

From this directory, launch Jupyter:

```bash
jupyter notebook
```

Open `assignment6.ipynb` and run the cells in order. The notebook downloads the SST-2 dataset from Hugging Face (`nyu-mll/glue`, `sst2`) and the pretrained model weights as needed. It saves processed datasets, trained models, tables, figures, and training checkpoints under the directories listed above.

The notebook includes BERT fine-tuning, optional hyperparameter experiments, DistilBERT and RoBERTa comparisons, attention visualizations, advanced training approaches, model export, and inference timing. Training and hyperparameter experiments can take significant time and benefit from a GPU; the multi-setting hyperparameter experiments are disabled by default.

The SST-2 test labels are hidden, so the notebook uses the validation split for evaluation and model selection.

## Run the Streamlit app

Run the notebook through the model fine-tuning step first so that `models/bert-finetuned/` is available. Then, from this directory, start the app:

```bash
streamlit run app.py
```

Enter a movie review and select **Predict** to see its positive or negative sentiment and confidence.

## Generated files and version control

Processed datasets, model weights, outputs, and training checkpoints are generated artifacts and are not included in the source commits. Run the notebook to recreate them.
