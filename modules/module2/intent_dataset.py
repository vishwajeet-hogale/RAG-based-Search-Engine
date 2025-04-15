import random
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import os

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "modules", "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")

# -----------------------------------
# STEP 1: Generate and Expand Dataset
# -----------------------------------

intent_examples = {
    "Labs": [
        "What labs are working on AI?",
        "Show me all labs related to cybersecurity",
        "Which lab focuses on human-computer interaction?",
        "Are there any robotics labs at Northeastern?",
        "List research labs in data science"
    ],
    "Research": [
        "What kind of research is happening in NLP?",
        "Give me recent research publications",
        "What is the latest research in machine learning?",
        "Find ongoing projects in the AI domain",
        "Tell me about research papers from Northeastern"
    ],
    "Professors": [
        "Who are the top professors in data science?",
        "Give me faculty profiles in computer vision",
        "Which professors are working on generative models?",
        "List all AI faculty members",
        "Tell me more about Professor Smith's research"
    ],
    "Institutes": [
        "What research centers are available in the college?",
        "List all institutes working on interdisciplinary science.",
        "Is there an institute for network science?",
        "Which institute focuses on cybersecurity?",
        "Tell me more about the Internet Democracy Initiative."
    ],
    "Research_current_highlights": [
        "What are the latest breakthroughs from the CS department?",
        "Show me current research highlights.",
        "Any recent achievements in robotics research?",
        "What’s new in the AI space at Northeastern?",
        "Give me recent news on faculty publications or awards."
    ]
}

# Add longer examples
long_examples = {
    "Labs": [
        "I'm interested in learning about the various research labs at Northeastern University that are involved in artificial intelligence and machine learning.",
    ],
    "Research": [
        "I'd like to know about the ongoing research projects in natural language processing across different departments.",
    ],
    "Professors": [
        "Who are the leading professors in the field of computer vision and where can I find their recent work?",
        "Who are the leading faculty in the field of computer vision and where can I find their recent work?"
    ],
    "Institutes": [
        "Tell me more about the institutes and interdisciplinary research centers associated with Northeastern University that work on AI.",
    ],
    "Research_current_highlights": [
        "What are the most recent and impactful discoveries made by the Khoury College research community?",
    ]
}

# Additional paraphrased/augmented entries
augment_phrases = [
    "Please help me with this:",
    "I'm curious to know,",
    "Could you assist with the following?",
    "Looking for info —",
    "As part of my research, I need to know:",
    "FYI,"
]

# Combine and boost dataset
dataset = []
for label, queries in {**intent_examples, **long_examples}.items():
    for text in queries:
        dataset.append({"text": text, "label": label})
    # Boost with paraphrased versions
    while len([d for d in dataset if d["label"] == label]) < 100:
        base = random.choice(queries)
        prefix = random.choice(augment_phrases)
        dataset.append({"text": f"{prefix} {base}", "label": label})

# Shuffle and split
random.shuffle(dataset)
df = pd.DataFrame(dataset)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Save CSVs
train_path = os.path.join(DATA_DIR,"intent_train_final.csv")
test_path = os.path.join(DATA_DIR,"intent_test_final.csv")
train_df.to_csv(train_path, index=False, quoting=csv.QUOTE_ALL)
test_df.to_csv(test_path, index=False, quoting=csv.QUOTE_ALL)

# train_path, test_path