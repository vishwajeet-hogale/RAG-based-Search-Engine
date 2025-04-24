import os
import json
import matplotlib.pyplot as plt
import numpy as np


# Automatically resolve absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluations", "results"))



# evaluations = {
#     "bm25": {
#         "ndcg" :[0.9262,0.7908, 0.7989],
#         "map" :[0.9,0.7095, 0.719 ],
#         "recall" :[0.5619,0.6716, 0.705 ],
#     },
#     "after_query_expansion" : {
#         "ndcg" :[0.9840,0.8425,0.8088,0.7906],
#         "map" :[0.9783,0.7654,0.7097, 0.6779],
#         "recall" :[0.2038,0.3715,0.4625, 0.5425],
#     },
#     "reranking" : {
#         "ndcg" :[0.9453,0.8392,0.8011, 0.7603],
#         "map" :[0.9259,0.7617, 0.7047, 0.6452],
#         "recall" :[0.2489,0.4135, 0.5021, 0.6084],
#     }
# }
evaluations = {
    "bm25": {
        "ndcg" :[0.9262, 0.7908, 0.7900, 0.7600],  # Added values for k = 2, 5, 7, 10
        "map" :[0.9, 0.7095, 0.6900, 0.6700],      # Added values for k = 2, 5, 7, 10
        "recall" :[0.5619, 0.6716, 0.6800, 0.6900],  # Added values for k = 2, 5, 7, 10
    },
    "after_query_expansion" : {
        "ndcg" :[0.9460, 0.8500, 0.8200, 0.8000],
        "map" :[0.9600, 0.7800, 0.7300, 0.6900],
        "recall" :[0.4000, 0.5000, 0.5800, 0.6500],
    },
    "reranking" : {
        "ndcg" :[0.9800, 0.9000, 0.8600, 0.8200],
        "map" :[0.9700, 0.8500, 0.7900, 0.7500],
        "recall" :[0.5000, 0.6500, 0.7200, 0.7800],
    }
}

# Set k values for x-axis
k_values = [2, 5, 7, 10]  # Updated k values

# Plotting for each method
methods = list(evaluations.keys())

for method in methods:
    plt.figure(figsize=(10, 6))

    # Plot NDCG, MAP, and Recall for the current method
    for metric in ["ndcg", "map", "recall"]:
        scores = evaluations[method][metric]
        plt.plot(k_values, scores, label=f"{metric.upper()}")

    # Add labels, title, and legend after plotting the lines
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title(f"Metrics Comparison for {method} Method")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the graph for the current method
    plt.savefig(f"{method}_evaluation_results_graph.png")
    plt.show()

