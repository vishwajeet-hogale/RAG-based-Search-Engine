# modules/query_expansion.py

import os
import torch
import chromadb
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")

# Load NLP & Model
nlp = spacy.load("en_core_web_lg")
device = "cuda" if torch.cuda.is_available() else "mps"
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1").to(device)

# Connect to ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DIR)
doc_collection = client.get_or_create_collection("research_index")
term_collection = client.get_or_create_collection("term_index")

# === spaCy tokenizer ===
def tokenize(text):
    return [token.text.lower() for token in nlp(text) if not token.is_punct and not token.is_space]

def expand_query(query, doc_top_k=5, term_top_k=5, max_final_expansions=5):
    print(f"\n🔍 Expanding query: {query}")
    query_embedding = model.encode(query, convert_to_numpy=True, device=device)
    query_tokens = set(tokenize(query))

    # === STEP 1: Retrieve top documents ===
    doc_results = doc_collection.query(
        query_embeddings=[query_embedding],
        n_results=doc_top_k,
        include=["documents"]
    )
    # print(f"Doc results: {doc_results["documents"][0]}")
    # Extract candidate terms from document text
    candidate_terms = set()
    for doc in doc_results["documents"][0]:
        if doc.strip():
            parsed = nlp(doc)
            candidate_terms.update(chunk.text.lower().strip() for chunk in parsed.noun_chunks)

    # === STEP 2: Remove terms found in original query ===
    filtered_candidates = [
        term for term in candidate_terms
        if term not in query.lower() and not any(tok in term for tok in query_tokens)
    ]

    if not filtered_candidates:
        print("⚠️ No valid candidate terms found in documents.")
        filtered_candidates = []

    # === STEP 3: Retrieve similar terms from term index ===
    term_results = term_collection.query(
        query_embeddings=[query_embedding],
        n_results=term_top_k,
        include=["documents"]
    )
    term_expansions = term_results["documents"][0]

    # === STEP 4: Combine & deduplicate terms ===
    combined_terms = list(set(term_expansions + filtered_candidates))

    if not combined_terms:
        print("⚠️ No expansion terms found.")
        return query, []

    # === STEP 5: Embed and rank final terms ===
    term_embeddings = model.encode(combined_terms, convert_to_numpy=True, device=device)
    similarities = cosine_similarity([query_embedding], term_embeddings)[0]
    top_indices = similarities.argsort()[-max_final_expansions:][::-1]
    final_expansions = [combined_terms[i] for i in top_indices]

    # === Final expanded query ===
    expanded_query = query + " " + " ".join(final_expansions)
    return expanded_query, final_expansions

# === CLI Test ===
if __name__ == '__main__':
    original_query = input("Enter your query: ")
    expanded_query, terms = expand_query(original_query)

    print("\n📝 Final Query Expansion:")
    print(f"Original Query   : {original_query}")
    print(f"Expansion Terms  : {terms}")
    print(f"Expanded Query   : {expanded_query}")
