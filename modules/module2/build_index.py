import os
import pandas as pd
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# === Universal path resolver ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "modules", "data")
CHROMA_DIR = os.path.join(PROJECT_ROOT, "vectorstore", "chroma")

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

# MAC
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# Windows
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Fix Keras/TF issue
os.environ["USE_TF"] = "0"

# Load SBERT
sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sbert_model = sbert_model.to(torch.device(device))

# Chroma client
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection("sbert_documents")

# Data sources
datasources = {
    "Labs": data_path("final_lab_summaries.csv"),
    "Research": data_path("final_research_info.csv"),
    "Professors": data_path("final_prof_details.csv"),
    "Institutes": data_path("institutes_and_centers.csv"),
    "Research_current_highlights": data_path("current_research_highlights.csv")
}

def build_vector_db():
    print("🔧 Building SBERT-based vector DB...")
    doc_counter = 0

    for name, path in datasources.items():
        if not os.path.exists(path):
            print(f"❌ Skipping {name}, file not found at {path}")
            continue

        df = pd.read_csv(path)

        if "doc_id" not in df.columns:
            df["doc_id"] = range(doc_counter, doc_counter + len(df))
            doc_counter += len(df)
            df.to_csv(path, index=False)

        print(f"Indexing datasource: {name}")
        for i, row in df.iterrows():
            print(f"  → Indexing doc {i}")
            doc_id = f"{name}_{row['doc_id']}"
            print(df.columns)
            text_parts = [str(row[col]) for col in df.columns
                          if col not in ["doc_id", "link", "url", "Publication Link", "Profile URL", "href"]]

            content = "\n".join(text_parts).strip()
            if not content:
                continue

            try:
                embedding = sbert_model.encode(content, convert_to_numpy=True, device=device)

                collection.add(
                    documents=[content],
                    ids=[doc_id],
                    metadatas=[{"source": name, "doc_id": row["doc_id"]}],
                    embeddings=[embedding]
                )
            except Exception as e:
                print(f"⚠️ Failed to embed doc {doc_id}: {e}")

    print("✅ SBERT Vector DB built.")

if __name__ == "__main__":
    build_vector_db()