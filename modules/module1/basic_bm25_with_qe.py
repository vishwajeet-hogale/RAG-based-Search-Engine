# import pandas as pd
# from rank_bm25 import BM25Okapi
# from nltk.tokenize import word_tokenize
# import nltk
# nltk.download("punkt_tab")

# # Load datasets
# def load_data():
#     professors = pd.read_csv("../data/final_prof_details.csv")
#     labs = pd.read_csv("../data/final_lab_summaries.csv")
#     institutes = pd.read_csv("../data/institutes_and_centers.csv")
#     research = pd.read_csv("../data/final_research_info.csv")
#     current_research_highlights = pd.read_csv("../data/current_research_highlights.csv")
#     return professors, labs, research, institutes, current_research_highlights

# # Preprocessing functions
# def preprocess_professors(df):
#     df["Text"] = (df["Professor Name"].fillna("") + " " +
#                    df["Research Area"].fillna("") + " " +
#                    df["Biography"].fillna("") + " " +
#                    df["Research Interests"].fillna("") + " " +
#                    df["doc_id"].fillna(""))
#     return df

# def preprocess_institutes(df):
#     df["Text"] = (df["name"].fillna("") + " " +
#                    df["description"].fillna("") + " " +
#                    df["doc_id"].fillna(""))
#     return df

# def preprocess_labs(df):
#     df["Text"] = (df["Research Area"].fillna("") + " " +
#                    df["Lab Name"].fillna("") + " " +
#                    df["Summary"].fillna("") + " " +
#                    df["doc_id"].fillna(""))
#     return df

# def preprocess_research(df):
#     df["Text"] = (df["Research Area"].fillna("") + " " +  
#                    df["Professor Name"].fillna("") + " " +
#                    df["Publication Title"].fillna("") + " " +
#                    df["Citation"].fillna("") + " " +
#                    df["Publication Summary"].fillna("") + " " +
#                    df["doc_id"])
#     return df

# def preprocess_curr_res(df):
#     df["Text"] = (df["title"].fillna("") + " " +  
#                    df["description"].fillna("") + " " +
#                    df["doc_id"].fillna(""))
#     return df

# # 🔹 Build BM25 Index
# def build_bm25(df):
#     tokenized = [word_tokenize(text.lower()) for text in df["Text"]]
#     return BM25Okapi(tokenized), tokenized

# # 🔹 Search Function
# def search(query, bm25_model, df, top_k=5):
#     tokens = word_tokenize(query.lower())
#     scores = bm25_model.get_scores(tokens)
#     top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
#     return df.iloc[top_indices], [scores[i] for i in top_indices]

# # 🔹 Integrated Search Function
# def integrated_search(query, top_k=5):
#     # Load and preprocess all datasets
#     prof_df, labs_df, res_df, ins_df, curr_df = load_data()
#     prof_df = preprocess_professors(prof_df)
#     labs_df = preprocess_labs(labs_df)
#     res_df = preprocess_research(res_df)
#     ins_df = preprocess_institutes(ins_df)
#     curr_df = preprocess_curr_res(curr_df)

#     # Build BM25 indices
#     prof_bm25, _ = build_bm25(prof_df)
#     labs_bm25, _ = build_bm25(labs_df)
#     res_bm25, _ = build_bm25(res_df)
#     ins_bm25, _ = build_bm25(ins_df)
#     curr_bm25, _ = build_bm25(curr_df)

#     # Perform searches
#     prof_results, prof_scores = search(query, prof_bm25, prof_df, top_k)
#     labs_results, labs_scores = search(query, labs_bm25, labs_df, top_k)
#     res_results, res_scores = search(query, res_bm25, res_df, top_k)
#     ins_results, ins_scores = search(query, ins_bm25, ins_df, top_k)
#     curr_results, curr_scores = search(query, curr_bm25, curr_df, top_k)

#     return {
#         "Professors": list(zip(prof_results.to_dict("records"), prof_scores)),
#         "Labs": list(zip(labs_results.to_dict("records"), labs_scores)),
#         "Research": list(zip(res_results.to_dict("records"), res_scores)),
#         "Institutes": list(zip(ins_results.to_dict("records"), ins_scores)),
#         "Current Research Highlights": list(zip(curr_results.to_dict("records"), curr_scores))
#     }

# if __name__ == "__main__":
#     query = input("🔍 Enter your search query: ")
#     results = integrated_search(query, top_k=5)

#     for section, items in results.items():
#         print(f"\n📚 Top {len(items)} results from {section}:")
#         for i, (row, score) in enumerate(items, start=1):
#             print(f"\n  ▶ Rank {i} (Score: {score:.2f})")
#             for key, value in row.items():
#                 print(f"    {key}: {value}")
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
nltk.download("punkt_tab")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "modules", "data")

# ===== Load CSVs =====
def load_data():
    paths = {
        "Professors": os.path.join(DATA_DIR,"final_prof_details.csv"),
        "Labs": os.path.join(DATA_DIR,"final_lab_summaries.csv"),
        "Research": os.path.join(DATA_DIR,"final_research_info.csv"),
        "Institutes": os.path.join(DATA_DIR,"institutes_and_centers.csv"),
        "Highlights": os.path.join(DATA_DIR,"current_research_highlights.csv")
    }
    return {name: pd.read_csv(path) for name, path in paths.items()}

# ===== Generic Preprocessing =====
def preprocess_text(df, cols):
    df["Text"] = df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    return df

# ===== Build BM25 Model =====
def build_bm25(df):
    tokenized = [word_tokenize(doc.lower()) for doc in df["Text"]]
    model = BM25Okapi(tokenized)
    return model, tokenized

# ===== Search Top-k =====
def search(query, model, df, top_k=5):
    tokens = word_tokenize(query.lower())
    scores = model.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return df.iloc[top_indices], [scores[i] for i in top_indices]

# ===== Unified Search Across Sources =====
def integrated_search(query, top_k=5):
    dfs = load_data()

    configs = {
        "Professors": ["Professor Name", "Research Area", "Biography", "Research Interests", "doc_id"],
        "Labs": ["Research Area", "Lab Name", "Summary", "doc_id"],
        "Research": ["Research Area", "Professor Name", "Publication Title", "Citation", "Publication Summary", "doc_id"],
        "Institutes": ["name", "description", "doc_id"],
        "Highlights": ["title", "description", "doc_id"]
    }

    results = {}
    for name, df in dfs.items():
        df = preprocess_text(df, configs[name])
        model, _ = build_bm25(df)
        top_df, top_scores = search(query, model, df, top_k)
        results[name] = list(zip(top_df.to_dict("records"), top_scores))

    return results

# ===== Run & Print =====
if __name__ == "__main__":
    query = input("🔍 Enter your search query: ").strip()
    results = integrated_search(query, top_k=5)

    for section, items in results.items():
        print(f"\n🔹 {section} — Top {len(items)} results:")
        for i, (row, score) in enumerate(items, start=1):
            print(f"\n  ▶ Rank {i} (Score: {score:.2f})")
            for key, val in row.items():
                if isinstance(val, str) and len(val) > 300:
                    val = val[:300] + "..."
                print(f"    {key}: {val}")
