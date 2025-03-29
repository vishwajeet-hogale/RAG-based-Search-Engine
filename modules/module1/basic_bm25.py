from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download("punkt_tab")

def get_publication_data(path="../data/"):
    return pd.read_csv(path + "professor_info.csv")

def preprocess_publication_text(df):
    # Combine title + citation
    df["Publication"] = df["Research Area"] + " " + df["Publication Title"].fillna("") + " " + df["Citation"].fillna("")
    return df

def build_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["Publication"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df, vectorizer

def build_bm25(df):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in df["Publication"]]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25

def search_bm25(bm25, query, df, top_k=5):
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return df.iloc[ranked_indices], [scores[i] for i in ranked_indices]

if __name__ == "__main__":
    df = get_publication_data()
    df = preprocess_publication_text(df)

    # Build TF-IDF (optional use elsewhere)
    tfidf_df, tfidf_vectorizer = build_tfidf(df)

    # Build BM25 using same text
    bm25_model = build_bm25(df)


    #### Basic version without 
    # Search with BM25
    query = "digital twins"
    results, scores = search_bm25(bm25_model, query, df, top_k=10)

    # Print results
    print("Top results for query:", query)
    for i, (index, row) in enumerate(results.iterrows()):
        print(f"\nRank {i+1} (Score: {scores[i]:.2f})")
        print("Title:", row["Publication Title"])
        print("Citation:", row["Citation"])
