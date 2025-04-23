import os
import json
import numpy as np
import pandas as pd
import sys
from transformers import pipeline

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.module2.roberta_query import search_expanded_query
# from modules.module2.intent_classifier import detect_intent



# ✅ Load the zero-shot classification model
classifier = pipeline("text-classification", model="./modules/module2/intent_model", tokenizer="./modules/module2/intent_model", device=0)

# ✅ Core function: detects intent from a single query
def detect_intent(query: str) -> str:
    result = classifier(query)
    print(result)
    result = result[0]
    print("Predicted intent:", result["label"], "Score:", result["score"])
    return result["label"]
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

def evaluate(k):
    with open("./evaluations/evaluations/results/crossencoder_ndcg_test_collection.json", "r") as f:
        ground_truth = json.load(f)

    ndcg_scores = []
    detailed_results = []
    ndcg = []

    for item in ground_truth:
        query = item["query"]
        relevance_dict = item["relevant_docs"]
        raw_results = search_expanded_query(query, top_k=45)
        # raw_results = raw_results[:k]

        # 🔧 Extract doc_ids from raw results (no reranking)
        metadatas = raw_results["metadatas"][0]  # List of metadata dicts
        retrieved_ids = [meta["doc_id"] for meta in metadatas[:k]]

        # 🔧 Get relevance scores from the cross-encoder-labeled ground truth
        relevances = [relevance_dict.get(str(doc_id), 0) for doc_id in retrieved_ids]

        score = ndcg_score(relevances)
        # if score > 0.0:
        ndcg.append(score)

        ndcg_scores.append((query, score))
        detailed_results.append({
            "query": query,
            "ndcg@10": round(score, 4)
        })

    # Save results
    os.makedirs("evaluations/results", exist_ok=True)
    with open("evaluations/results/evaluation_raw_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    avg_score = sum(ndcg) / len(ndcg) if ndcg else 0.0
    print(f"\n✅ Evaluation (raw results) completed — Avg NDCG@10: {round(avg_score, 4)}")



if __name__ == "__main__":
    evaluate(20)
