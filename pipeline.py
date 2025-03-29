from scrapers import scraper
from modules.module2.build_index import build_vector_db
import pandas as pd
import os

# === Universal path setup ===
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "modules", "data")

def data_path(filename):
    return os.path.join(DATA_DIR, filename)

# Data sources
datasources = {
    "Labs": data_path("labs_with_summaries.csv"),
    "Research": data_path("professor_info.csv"),
    "Professors": data_path("professors.csv"),
    "Institutes": data_path("research_institutes.csv"),
    "Research_current_highlights": data_path("current_research_highlights.csv")
}

if __name__ == "__main__":
    print("🔄 Running scraper...")
    scraper.main()

    print("🧾 Assigning global document IDs...")
    doc_counter = 0

    for name, path in datasources.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            num_rows = len(df)
            df["doc_id"] = range(doc_counter, doc_counter + num_rows)
            doc_counter += num_rows
            df.to_csv(path, index=False)
            print(f"✅ {name}: {num_rows} docs assigned.")
        else:
            print(f"⚠️ Warning: {name} file not found at {path}")

    build_vector_db()
    