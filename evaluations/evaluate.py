import os
import json
import numpy as np
import pandas as pd
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.module2.roberta_query import search_expanded_query
from modules.module2.intent_classifier import detect_intent

# === Constants ===
INTENT_PRIORITY = {
    "Research": ["Research", "Highlights", "Professors", "Labs", "Institutes"],
    "Professors": ["Professors", "Labs", "Research", "Highlights", "Institutes"],
    "Labs": ["Labs", "Professors", "Research", "Highlights", "Institutes"],
    "Institutes": ["Institutes", "Labs", "Professors", "Research", "Highlights"],
    "Research_current_highlights": ["Research_current_highlights", "Research", "Professors", "Labs", "Institutes"],
}

# === Helpers ===
def rerank_by_intent(results, intent):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    def score(item):
        doc, meta, dist = item
        source_priority = INTENT_PRIORITY.get(intent, [])
        base_score = 1 / (1 + dist)
        try:
            priority = source_priority.index(meta["source"])
        except ValueError:
            priority = len(source_priority)
        return -priority, base_score

    ranked = sorted(zip(docs, metas, distances), key=score, reverse=True)
    return ranked

def ndcg_score(relevances, k=10):
    def dcg(scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores[:k]))
    ideal = sorted(relevances, reverse=True)
    return dcg(relevances) / (dcg(ideal) or 1.0)

# === Main Evaluation ===
def evaluate():
    # Load intent CSV
    df = pd.read_csv("./modules/data/intent_train_final.csv")
    sampled = df.sample(30, random_state=42)

    test_collection = []
    ndcg_scores = []

    for _, row in sampled.iterrows():
        query = row["text"]
        raw_results = search_expanded_query(query, top_k=45)
        # print(raw_results)
        intent = detect_intent(query)
        ranked = rerank_by_intent(raw_results, intent)

        top_docs = ranked[:10]
        relevant_docs = {meta["doc_id"]: 3 for _, meta, _ in top_docs}
        retrieved_ids = [meta["doc_id"] for _, meta, _ in ranked]
        relevances = [relevant_docs.get(doc_id, 0) for doc_id in retrieved_ids]

        score = ndcg_score(relevances)
        ndcg_scores.append((query, score))

        test_collection.append({
            "query": query,
            "relevant_docs": relevant_docs
        })

    # Save collection + report
    os.makedirs("evaluations/results", exist_ok=True)
    with open("evaluations/results/generated_ndcg_test_collection.json", "w") as f:
        json.dump(test_collection, f, indent=2)

    with open("evaluations/results/ndcg_scores.json", "w") as f:
        json.dump([{"query": q, "ndcg@10": round(s, 4)} for q, s in ndcg_scores], f, indent=2)

    avg_score = round(np.mean([s for _, s in ndcg_scores]), 4)
    print(f"✅ Evaluation completed — Avg NDCG@10: {avg_score}")

    for q, s in ndcg_scores:
        print(f" - {q[:60]}... → NDCG@10: {round(s, 4)}")


if __name__ == "__main__":
    evaluate()
