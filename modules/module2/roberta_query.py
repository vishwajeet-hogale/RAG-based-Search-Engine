# modules/query_and_search.py

import os
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from roberta_qe import expand_query
from intent_classifier import detect_intent

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-roberta-large-v1").to(device)

# === ChromaDB client & collection ===
client = chromadb.PersistentClient(path=CHROMA_DIR)
doc_collection = client.get_or_create_collection("research_index")

def search_expanded_query(query, top_k=15):
    # Step 1: Expand the query
    expanded_query, expansion_terms = expand_query(query)

    print("\n🔎 Expanded Query:")
    print(f"Original : {query}")
    print(f"Expanded : {expanded_query}")
    print(f"Terms    : {expansion_terms}")

    # Step 2: Embed the expanded query
    query_embedding = model.encode(expanded_query, convert_to_numpy=True, device=device)

    # Step 3: Query the vector index
    results = doc_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results

def print_ranked_results(results,preferred_type=None):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    scored = []
    # Convert to score and sort
    if preferred_type:
        # boost top-ranked items from the preferred source
        scored = sorted(zip(docs, metas, distances), key=lambda x: (x[1]["source"] == preferred_type, 1 / (1 + x[2])), reverse=True)
    else:
        scored = sorted(
            zip(docs, metas, distances),
            key=lambda x: 1 / (1 + x[2]),  # Higher score = more relevant
            reverse=True
        )

    print("\n📊 Top Results (Ranked by Semantic Score):")
    for i, (doc, meta, dist) in enumerate(scored, 1):
        score = round(1 / (1 + dist), 4)
        print(f"\n🔹 Rank #{i} — Score: {score}")
        print(f"Metadata: {meta}")
        print(f"Content Preview: {doc[:500]}...")
        print("-" * 80)

# === CLI entry point ===
if __name__ == '__main__':
    while True:
        query = input("\nEnter your query (or type 'quit' to exit): ").strip()
        if query.lower() in {"quit", "exit"}:
            print("👋 Exiting. Goodbye!")
            break
        results = search_expanded_query(query)
        preferred_type = detect_intent(query)
        # print(f"Intent: {preferred_type}")
        print_ranked_results(results,preferred_type)
