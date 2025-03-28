from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import os

nltk.download("punkt")

def get_publication_data(path="../data/"):
    return pd.read_csv(path + "professor_info.csv")

def preprocess_publication_text(df):
    # Combine title + citation
    df["Publication"] = df["Research Area"].fillna("") + " " + df["Publication Title"].fillna("") + " " + df["Citation"].fillna("")
    return df

def build_tfidf(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df["Publication"])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return tfidf_df, vectorizer

def build_bm25(df):
    tokenized_docs = [word_tokenize(doc.lower()) for doc in df["Publication"]]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs

def train_word2vec(tokenized_docs, model_path="word2vec.model"):
    if os.path.exists(model_path):
        model = Word2Vec.load(model_path)
    else:
        model = Word2Vec(sentences=tokenized_docs, vector_size=100, window=5, min_count=2, workers=4)
        model.save(model_path)
    return model

def expand_query(query, w2v_model, topn=3):
    query_tokens = word_tokenize(query.lower())
    expanded_query = query_tokens.copy()

    for token in query_tokens:
        if token in w2v_model.wv:
            similar_words = [word for word, sim in w2v_model.wv.most_similar(token, topn=topn)]
            expanded_query.extend(similar_words)
    print(expanded_query)
    return expanded_query

def search_bm25(bm25, query, df, top_k=5, w2v_model=None, expand=False, topn=3):
    if expand and w2v_model:
        tokenized_query = expand_query(query, w2v_model, topn=topn)
        
    else:
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
    bm25_model, tokenized_docs = build_bm25(df)

    # Train/load Word2Vec model
    w2v_model = train_word2vec(tokenized_docs)

    # Search with expanded query
    query = "deep learning accelerator unit"
    results, scores = search_bm25(bm25_model, query, df, top_k=10, w2v_model=w2v_model, expand=True, topn=10)

    print("Top results for expanded query:", query)
    for i, (index, row) in enumerate(results.iterrows()):
        print(f"\nRank {i+1} (Score: {scores[i]:.2f})")
        print("Title:", row["Publication Title"])
        print("Citation:", row["Citation"])
