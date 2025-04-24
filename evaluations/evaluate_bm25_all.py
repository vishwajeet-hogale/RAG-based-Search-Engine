import os
import json
import numpy as np
import sys
import nltk

nltk.download('punkt_tab')

# Add module path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules.module1.basic_bm25_with_qe import integrated_search

# -------------------------------
# Scoring Functions
# -------------------------------
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

# -------------------------------
# Evaluation Runner
# -------------------------------
def evaluate(k):
    print(f"\n🚀 Starting evaluation with top-{k} documents per query...\n")

    try:
        with open("./evaluations/results/crossencoder_ndcg_test_collection.json", "r") as f:
            ground_truth = json.load(f)
    except FileNotFoundError:
        print("❌ Error: Ground truth file not found.")
        return

    ndcg = []
    map_scores = []
    recall_scores = []
    detailed_results = []

    for idx, item in enumerate(ground_truth):
        query = item["query"]
        relevance_dict = item["relevant_docs"]  # {doc_id: score}

        print(f"\n🔍 [{idx+1}/{len(ground_truth)}] Processing query: {query}")

        try:
            results = integrated_search(query, top_k=45)
        except Exception as e:
            print(f"❌ Error running integrated search for query: {query}\n   {e}")
            continue

        flat_results = []
        for source, records in results.items():
            for doc, score in records:
                doc_id = str(doc.get("doc_id", ""))
                flat_results.append((doc_id, score))

        flat_results.sort(key=lambda x: x[1], reverse=True)
        retrieved_ids = [doc_id for doc_id, _ in flat_results[:k]]
        relevances = [relevance_dict.get(doc_id, 0) for doc_id in retrieved_ids]

        print(f"Retrieved doc_ids: {retrieved_ids}")
        print(f"Relevant docs: {relevance_dict.keys()}")
        print(f"Matched Relevant Docs: {[doc_id for doc_id in retrieved_ids if doc_id in relevance_dict]}")

        score_ndcg = ndcg_score(relevances, k)
        score_map = average_precision_at_k(relevances, k)
        score_recall = recall_at_k(relevances, len(relevance_dict), k)

        if score_ndcg > 0.0:
            ndcg.append(score_ndcg)
        if score_map > 0.0:
            map_scores.append(score_map)
        if score_recall > 0.0:
            recall_scores.append(score_recall)

        print(f"   ✅ NDCG@{k}: {round(score_ndcg, 4)} | MAP@{k}: {round(score_map, 4)} | Recall@{k}: {round(score_recall, 4)}")

        detailed_results.append({
            "query": query,
            "ndcg@10": round(score_ndcg, 4),
            "map@10": round(score_map, 4),
            "recall@10": round(score_recall, 4),
            "top_docs": retrieved_ids,
            "relevance_scores": relevances
        })

    # Save results
    os.makedirs("./evaluations/results", exist_ok=True)
    output_path = "./evaluations/results/evaluation_basic_bm25_qe.json"
    with open(output_path, "w") as f:
        json.dump(detailed_results, f, indent=2)

    avg_ndcg = sum(ndcg) / len(ndcg) if ndcg else 0.0
    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    print(f"\n✅ Evaluation completed — Avg NDCG@{k}: {round(avg_ndcg, 4)} | Avg MAP@{k}: {round(avg_map, 4)} | Avg Recall@{k}: {round(avg_recall, 4)}")
    print(f"📁 Results saved to: {output_path}")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    evaluate(10)
