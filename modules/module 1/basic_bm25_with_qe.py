import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")

# Load datasets
def load_data():
    professors = pd.read_csv("../data/professor_info.csv")
    labs = pd.read_csv("../data/research_institutes.csv")
    research = pd.read_csv("../data/professor_info.csv")
    return professors, labs, research

# Preprocessing functions
def preprocess_professors(df):
    df["Text"] = (df["Professor Name"].fillna("") + " " +
                   df["Research Interests"].fillna("") + " " +
                   df["Biography"].fillna("") + " " +
                   df["Education"].fillna("") + " " +
                   df["Research Area"].fillna(""))
    return df

def preprocess_labs(df):
    df["Text"] = (df["Institute Name"].fillna("") + " " +
                   df["Description"].fillna(""))
    return df

def preprocess_research(df):
    df["Text"] = (df["Publication Title"].fillna("") + " " +
                   df["Citation"].fillna("") + " " +
                   df["Research Area"].fillna(""))
    return df

# Build BM25
def build_bm25(df):
    tokenized = [word_tokenize(text.lower()) for text in df["Text"]]
    return BM25Okapi(tokenized), tokenized

# Search function
def search(query, bm25_model, df, top_k=5):
    tokens = word_tokenize(query.lower())
    scores = bm25_model.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return df.iloc[top_indices], [scores[i] for i in top_indices]

# Main integrated search
def integrated_search(query, top_k=5):
    prof_df, labs_df, res_df = load_data()
    prof_df = preprocess_professors(prof_df)
    labs_df = preprocess_labs(labs_df)
    res_df = preprocess_research(res_df)

    prof_bm25, _ = build_bm25(prof_df)
    labs_bm25, _ = build_bm25(labs_df)
    res_bm25, _ = build_bm25(res_df)

    prof_results, prof_scores = search(query, prof_bm25, prof_df, top_k)
    labs_results, labs_scores = search(query, labs_bm25, labs_df, top_k)
    res_results, res_scores = search(query, res_bm25, res_df, top_k)

    return {
        "Professors": list(zip(prof_results[["Professor Name", "Biography", "Education"]].to_dict("records"), prof_scores)),
        "Labs": list(zip(labs_results[["Institute Name", "Description"]].to_dict("records"), labs_scores)),
        "Research": list(zip(res_results[["Professor Name", "Publication Title", "Citation"]].to_dict("records"), res_scores))
    }

# Example usage
if __name__ == "__main__":
    query = "feedback loops"
    results = integrated_search(query, top_k=5)

    for category, items in results.items():
        print(f"\n--- Top {len(items)} {category} ---")
        for i, (item, score) in enumerate(items):
            print(f"\nRank {i+1} (Score: {score:.2f})")
            for k, v in item.items():
                print(f"{k}: {v}")