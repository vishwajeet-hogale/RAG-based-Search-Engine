from transformers import pipeline
import torch
import os

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "modules", "models")

# ✅ Load on GPU if available
device = "cuda" if torch.cuda.is_available() else "mps"

# ✅ Load the zero-shot classification model
classifier = pipeline("text-classification", model="./modules/module2/intent_model", tokenizer="./modules/module2/intent_model", device=0)

# ✅ Core function: detects intent from a single query
def detect_intent(query: str) -> str:
    result = classifier(query)
    print(result)
    result = result[0]
    print("Predicted intent:", result["label"], "Score:", result["score"])
    return result["label"]

# ✅ Optional: CLI usage for testing
if __name__ == "__main__":
    while True:
        user_input = input("🧠 Enter a query (or 'q' to quit): ")
        if user_input.lower() == "q":
            break
        intent = detect_intent(user_input)
        # print(f"🎯 Predicted intent: {intent}\n")
