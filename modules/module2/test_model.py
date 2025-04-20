import pandas as pd
import os
import torch
from transformers import pipeline
from intent_classifier import detect_intent

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "modules", "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")
MODELS_DIR = os.path.join(PROJECT_ROOT, "modules", "models")

device = "cuda" if torch.cuda.is_available() else "mps"
# Load your fine-tuned intent classifier
classifier = pipeline(
    "text-classification",
    model="./intent_model",         # or the path where you saved it
    tokenizer="./intent_model",
    device=device  # use CUDA if available, else set to -1 for CPU
)

df = pd.read_csv(os.path.join(DATA_DIR,"intent_test_final.csv"))  # Column: text
df["prediction"] = df["text"].apply(lambda q: classifier(q)[0]['label'])
df["confidence"] = df["text"].apply(lambda q: classifier(q)[0]['score'])

df.to_csv(os.path.join(DATA_DIR,"predicted_intents.csv"), index=False)
print("✅ Predictions saved to predicted_intents.csv")