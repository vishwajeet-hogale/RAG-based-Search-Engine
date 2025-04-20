# modules/index_creation.py

import os
import pandas as pd
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import spacy

# === Paths ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "modules", "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")

# === Helper ===
def data_path(filename):
    return os.path.join(DATA_DIR, filename)

# === Sources ===
datasources = {
    "Labs": data_path("final_lab_summaries.csv"),
    "Research": data_path("final_research_info.csv"),
    "Professors": data_path("final_prof_details.csv"),
    "Institutes": data_path("institutes_and_centers.csv"),
    "Research_current_highlights": data_path("current_research_highlights.csv")
}

EXCLUDE_COLS = {"doc_id", "link", "url", "Publication Link", "Profile URL", "href"}

# === Batching for ChromaDB ===
def batch_upsert(collection, documents, embeddings, ids, metadatas, batch_size=5000):
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            embeddings=embeddings[i:i+batch_size],
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

def build_indexes():
    print("🚀 Building document and term indexes...")

    device = "cuda" if torch.cuda.is_available() else "mps"
    model = SentenceTransformer("sentence-transformers/all-roberta-large-v1").to(device)
    nlp = spacy.load("en_core_web_lg")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    client.delete_collection("research_index")
    client.delete_collection("term_index")
    doc_collection = client.get_or_create_collection("research_index")
    term_collection = client.get_or_create_collection("term_index")

    all_texts = []
    all_sources = []
    all_ids = []
    all_metadatas = []

    doc_id_counter = 0  # Global doc_id

    print("📄 Processing data sources...")

    for source, path in datasources.items():
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue

        df = pd.read_csv(path)
        print(f"✅ Loaded {source}: {len(df)} rows")

        if "doc_id" not in df.columns:
            df["doc_id"] = range(doc_id_counter, doc_id_counter + len(df))
            doc_id_counter += len(df)
        else:
            # Ensure unique global doc_id
            offset = doc_id_counter
            df["doc_id"] = df["doc_id"].apply(lambda x: x + offset)
            doc_id_counter += len(df)

        for _, row in df.iterrows():
            parts = [
                str(row[col]).strip() for col in df.columns
                if col not in EXCLUDE_COLS and pd.notna(row[col])
            ]
            combined_text = " ".join(parts).strip()

            if not combined_text:
                continue

            all_texts.append(combined_text)
            all_sources.append(source)
            all_ids.append(str(row["doc_id"]))
            all_metadatas.append({"source": source, "doc_id": row["doc_id"]})

    print(f"🧠 Embedding {len(all_texts)} documents...")
    embeddings = model.encode(all_texts, convert_to_numpy=True, device=device)

    print(f"📦 Storing documents in vector DB...")
    batch_upsert(doc_collection, all_texts, embeddings.tolist(), all_ids, all_metadatas)

    # === Term Indexing ===
    print("🧩 Extracting and indexing terms...")
    term_set = set()
    for text in all_texts:
        doc = nlp(text)
        term_set.update(chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.text.strip())

    terms = list(term_set)
    term_ids = [f"term_{i}" for i in range(len(terms))]
    term_embeddings = model.encode(terms, convert_to_numpy=True, device=device)
    term_metadatas = [{"term": t} for t in terms]
    print(f"🔁 Storing {len(terms)} terms in term index...")
    batch_upsert(term_collection, terms, term_embeddings.tolist(), term_ids, term_metadatas)

    print("✅ Indexing complete.")

if __name__ == '__main__':
    build_indexes()
