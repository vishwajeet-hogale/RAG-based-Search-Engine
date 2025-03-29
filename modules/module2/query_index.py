# modules/module2/query_index.py

import os
import chromadb
from sentence_transformers import SentenceTransformer
import torch

# === Universal path setup ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")

# Load ChromaDB persistent collection
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("sbert_documents")

# Use MPS or fallback
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Use SBERT
os.environ["USE_TF"] = "0"
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sbert_model = sbert_model.to(torch.device(device))

# === Query Function ===
def query_vector_db(query_text, top_k=5):
    print(f"\n🔍 Querying vector DB for: '{query_text}'")

    embedding = sbert_model.encode(query_text, convert_to_numpy=True, device=device)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["distances", "metadatas", "documents"]
    )

    hits = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = 1 / (1 + dist)  # or: 1 - dist if using cosine distance
        hits.append({
            "source": meta["source"],
            "doc_id": meta["doc_id"],
            "content": doc,
            "distance": dist,
            "score": round(score, 4)
        })

    return hits

# === CLI Entry Point (optional) ===
if __name__ == "__main__":
    user_query = "Show me research in the field of NLP"
    results = query_vector_db(user_query, top_k=7)



    print("\n📝 Top Matches:")
    for i, res in enumerate(results):
        print(f"\n[{i+1}] Source: {res['source']}, Doc ID: {res['doc_id']}\n{res['content']}\n")
