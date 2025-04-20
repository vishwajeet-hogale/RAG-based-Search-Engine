#!/bin/bash

echo "🔧 Creating virtual environment..."
python3 -m venv .venv_retrieval
source .venv_retrieval/bin/activate

echo "⬆️ Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "📦 Downloading spaCy model..."
python -m spacy download en_core_web_lg

echo "🧠 Training intent classifier..."
python modules/module2/train_intent_classifier.py

echo "🧱 Building Roberta-based index..."
python modules/module2/roberta_index.py

echo "🚀 Launching Streamlit app..."
streamlit run final_app.py
