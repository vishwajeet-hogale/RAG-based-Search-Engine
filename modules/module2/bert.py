
"""
import json
import re
import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk

nltk.download('punkt')

# Load SBERT model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Read the data
df = pd.read_csv("/Users/User/Documents/IR/RAG-based-Search-Engine/modules/data/professor_details.csv")

# Clean the text: remove non-alphabetic characters and numbers
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower()

# Create the corpus from selected columns
df['corpus'] = (df['Professor Name'].fillna('') + ' ' + df['Publication Title'].fillna(''))

# Apply text cleaning
df['corpus'] = df['corpus'].apply(clean_text)

# Tokenize the corpus into unigrams
unigrams = set()
for text in df['corpus']:
    unigrams.update(nltk.word_tokenize(text))

unigrams = list(unigrams)  # Convert to list

# Generate embeddings for unigrams
unigram_embeddings = model.encode(unigrams, convert_to_tensor=False)  # Use NumPy array

# Create FAISS index for fast similarity search
d = unigram_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(unigram_embeddings)

# Function to find similar words using FAISS
def find_similar_words(word_list, word_embeddings, threshold=0.8, top_k=5):
    similar_words_dict = {}

    for idx, word in enumerate(word_list):
        query_embedding = np.expand_dims(word_embeddings[idx], axis=0)  # Reshape for FAISS
        _, similar_indices = index.search(query_embedding, top_k)  # Get top-k similar words

        similar_words = [
            word_list[i] for i in similar_indices[0] if i != idx  # Exclude itself
        ]
        similar_words_dict[word] = similar_words

    return similar_words_dict

# Find similar words
similar_words_data = find_similar_words(unigrams, unigram_embeddings, threshold=0.8)

# Save to JSON
with open("unigram_expanded.json", "w") as f:
    json.dump(similar_words_data, f, indent=4)

print("Unigram-based expansion saved to unigram_expanded.json ✅")
"""

import json
import re
import torch
import faiss
import numpy as np
import pandas as pd
import nltk
import spacy
from sentence_transformers import SentenceTransformer

# Download necessary resources
nltk.download("punkt")

# Load spaCy NLP model
nlp = spacy.load("en_core_web_sm")

# Load SBERT model
model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Read the data
df = pd.read_csv("/Users/User/Documents/IR/RAG-based-Search-Engine/modules/data/professor_details.csv")

# Combine relevant columns to create the text corpus
df['corpus'] = (df['Professor Name'].fillna('') + ' ' + df['Publication Title'].fillna(''))

# Function to clean text: remove non-alphabetic characters, convert to lowercase
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove numbers & special characters
    text = text.lower()
    return text

# Function to remove stopwords using spaCy
def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop and token.is_alpha])

# Apply cleaning and stopword removal
df['corpus'] = df['corpus'].apply(clean_text).apply(remove_stopwords)

# Function to extract unigrams (single words)
def get_unigrams(text):
    doc = [token.text for token in nlp(text) if token.is_alpha]  # Only keep alphabetic tokens
    return set(doc)  # Return unigrams as a set to avoid duplicates

# Extract unigrams from the corpus
unigrams = set()
for text in df['corpus']:
    unigrams.update(get_unigrams(text))

unigrams = list(unigrams)  # Convert to list

# Generate embeddings for unigrams
unigram_embeddings = model.encode(unigrams, convert_to_tensor=False)  # Use NumPy array

# Create FAISS index for fast similarity search
d = unigram_embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)
index.add(unigram_embeddings)

# Function to find similar words using FAISS
def find_similar_words(word_list, word_embeddings, threshold=0.8, top_k=5):
    similar_words_dict = {}

    for idx, word in enumerate(word_list):
        query_embedding = np.expand_dims(word_embeddings[idx], axis=0)  # Reshape for FAISS
        _, similar_indices = index.search(query_embedding, top_k)  # Get top-k similar words

        similar_words = [
            word_list[i] for i in similar_indices[0] if i != idx  # Exclude itself
        ]
        similar_words_dict[word] = similar_words

    return similar_words_dict

# Find similar words
similar_words_data = find_similar_words(unigrams, unigram_embeddings, threshold=0.8)

# Save to JSON
output_file = "unigram_expanded.json"
with open(output_file, "w") as f:
    json.dump(similar_words_data, f, indent=4)

print(f"Unigram-based expansion saved to {output_file} ✅")
