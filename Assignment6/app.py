import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./models/bert-finetuned"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

st.title("Movie Review Sentiment Analysis")
st.write("Enter a sentence to predict whether its sentiment is positive or negative.")

tokenizer, model, device = load_model()
text = st.text_area("Enter your review")

if st.button("Predict"):
    if text.strip():
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)[0]

        predicted_class = int(torch.argmax(probabilities).item())
        sentiment = "Positive" if predicted_class == 1 else "Negative"
        confidence = float(probabilities[predicted_class].item()) * 100

        st.subheader("Prediction: " + sentiment)
        st.write("Confidence: {:.2f}%".format(confidence))
        st.progress(confidence / 100)
    else:
        st.warning("Please enter a sentence.")
