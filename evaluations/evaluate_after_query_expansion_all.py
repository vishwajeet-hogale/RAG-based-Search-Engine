import os
import json
import numpy as np
import pandas as pd
import sys
from transformers import pipeline

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.module2.roberta_query import search_expanded_query

def ndcg_score(relevances, k=10):
    def dcg(scores):
        return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(scores[:k]))
    ideal = sorted(relevances, reverse=True)
    return dcg(relevances) / (dcg(ideal) or 1.0)

def average_precision_at_k(relevances, k=10):
    num_hits = 0.0
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            num_hits += 1.0
            score += num_hits / (i + 1)
    return score / max(1, sum(1 for rel in relevances[:k] if rel > 0))

def recall_at_k(relevances, total_relevant, k=10):
    retrieved_relevant = sum(1 for rel in relevances[:k] if rel > 0)
    return retrieved_relevant / max(1, total_relevant)

def evaluate(k):
    with open("./evaluations/evaluations/results/crossencoder_ndcg_test_collection.json", "r") as f:
        ground_truth = json.load(f)

    ndcg = []
    map_scores = []
    recall_scores = []
    detailed_results = []

    for item in ground_truth:
        query = item["query"]
        relevance_dict = item["relevant_docs"]
        raw_results = search_expanded_query(query, top_k=45)

        metadatas = raw_results["metadatas"][0]
        retrieved_ids = [meta["doc_id"] for meta in metadatas[:k]]
        relevances = [relevance_dict.get(str(doc_id), 0) for doc_id in retrieved_ids]

        score_ndcg = ndcg_score(relevances, k)
        score_map = average_precision_at_k(relevances, k)
        score_recall = recall_at_k(relevances, len(relevance_dict), k)

        if score_ndcg > 0.0:
            ndcg.append(score_ndcg)
        if score_map > 0.0:
            map_scores.append(score_map)
        if score_recall > 0.0:
            recall_scores.append(score_recall)

        detailed_results.append({
            "query": query,
            "ndcg@10": round(score_ndcg, 4),
            "map@10": round(score_map, 4),
            "recall@10": round(score_recall, 4)
        })
        print({
            "query": query,
            "ndcg@10": round(score_ndcg, 4),
            "map@10": round(score_map, 4),
            "recall@10": round(score_recall, 4)
        })

    os.makedirs("evaluations/results", exist_ok=True)
    with open("evaluations/results/evaluation_raw_results.json", "w") as f:
        json.dump(detailed_results, f, indent=2)

    avg_ndcg = sum(ndcg) / len(ndcg) if ndcg else 0.0
    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    print("No. of queries : ", len(ndcg))
    print(f"\n✅ Evaluation (raw results) completed — Avg NDCG@10: {avg_ndcg:.4f} | Avg MAP@10: {avg_map:.4f} | Avg Recall@10: {avg_recall:.4f}")
    return detailed_results

if __name__ == "__main__":
    evaluate(10)
