import os
import json
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import CrossEncoder

# Automatically use GPU if available
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

# Load cross-encoder model
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

# Load datasets
labs = pd.read_csv("final_lab_summaries.csv")
profs = pd.read_csv("final_prof_details.csv")
research = pd.read_csv("final_research_info.csv")
institutes = pd.read_csv("institutes_and_centers.csv")
highlights = pd.read_csv("current_research_highlights.csv")

# Query-intent mapping
query_intent_map = {
    "What are some key research highlights at Northeastern?": "Research_current_highlights",
    "Tell me about active labs at Northeastern University.": "Labs",
    "Which labs are working on data systems and data management?": "Labs",
    "What research is being done in robotics and human interaction?": "Labs",
    "Who is studying computational modeling of human emotions?": "Labs",
    "Are there any research projects that received major grants?": "Research",
    "Which professors are working on perception and planning?": "Professors",
    "Tell me about research in sustainable risk reduction.": "Research",
    "What is the focus of the Helping Hands Lab?": "Labs",
    "Is anyone using alternate reality games in research?": "Research",
    "What are the latest publications in computer vision?": "Research",
    "Who is working on nonverbal behavior in social interactions?": "Labs",
    "Which institutes work on human-centered computing?": "Institutes",
    "Are there research centers for smart city technologies?": "Institutes",
    "What is the CESAR lab currently researching?": "Labs",
    "List labs focused on decision-making and emotion modeling.": "Labs",
    "Any faculty researching intelligent systems at Northeastern?": "Professors",
    "Tell me about interdisciplinary research centers.": "Institutes",
    "Who works on uncertainty and inference over networks?": "Research",
    "Any labs involved in AI and education technologies?": "Labs",
    "Are there labs on healthcare data analytics?": "Labs",
    "Give me professors researching large-scale data lakes.": "Professors",
    "What robotics labs work with humans in real-world environments?": "Labs",
    "Who is researching scalable inference methods?": "Research",
    "What kind of research funding has Northeastern received?": "Research",
    "Where is research on community resilience being done?": "Research",
    "What kind of grants are awarded for risk analysis work?": "Research",
    "Are there research efforts involving public policy?": "Research",
    "Give me highlights from Northeastern's data science work.": "Research",
    "List key collaborations between labs and institutions.": "Labs",
    "Who is leading the research on autonomous systems?": "Professors",
    "What are the newest breakthroughs in AI at Northeastern?": "Research_current_highlights",
    "Tell me about labs focused on robotics and machine learning.": "Labs",
    "Are there any interdisciplinary projects involving healthcare and AI?": "Research",
    "Which institutes deal with urban sustainability and resilience?": "Institutes",
    "Who are the key researchers in computer security?": "Professors",
    "What labs study human-computer interaction at Khoury College?": "Labs",
    "Which research centers focus on ethical computing?": "Institutes",
    "Are there professors working on explainable AI?": "Professors",
    "Tell me about research on misinformation and social networks.": "Research",
    "What are some recent funded projects in data science?": "Research",
    "Which professors have expertise in computational linguistics?": "Professors",
    "What labs are doing work on autonomous vehicles?": "Labs",
    "List institutes researching technology and society.": "Institutes",
    "Is there a lab focused on affective computing?": "Labs",
    "What research is happening at the interface of law and tech?": "Research",
    "Who is investigating multi-agent systems?": "Professors",
    "Which centers collaborate on AI policy development?": "Institutes",
    "Are there research efforts in climate tech?": "Research",
    "List professors researching fairness in machine learning.": "Professors",
    "What are some recent AI-focused research highlights?": "Research_current_highlights",
    "Tell me about research on learning analytics.": "Research",
    "Which labs are involved in virtual and augmented reality?": "Labs",
    "Who is working on computational social science?": "Professors",
    "Are there active research centers in cybersecurity policy?": "Institutes",
    "What faculty are working on visual recognition systems?": "Professors",
    "What are some recent interdisciplinary research highlights?": "Research_current_highlights",
    "Tell me about faculty involved in human-centered AI.": "Professors",
    "Are there labs focused on data visualization?": "Labs",
    "Which institute specializes in AI and ethics?": "Institutes",
    "Which professors are publishing in neural networks?": "Professors",
    "Are there any labs doing work on smart infrastructure?": "Labs",
    "What recent research has Northeastern done in generative AI?": "Research_current_highlights",
    "Which centers study human factors in computing?": "Institutes",
    "Tell me about interdisciplinary work between health and computing.": "Research",
    "Any labs applying AI to social good?": "Labs",
    "List professors researching probabilistic programming.": "Professors",
    "What institutes focus on responsible technology?": "Institutes",
    "What are the latest innovations in Khoury's AI research?": "Research_current_highlights",
    "Which labs study cognitive systems?": "Labs",
    "Are there research initiatives on algorithmic bias?": "Research",
    "Who is working on emotion-aware AI systems?": "Professors",
    "Which institutes collaborate with external industry partners?": "Institutes",
    "What labs are developing learning agents?": "Labs",
    "Are there new breakthroughs in natural language generation?": "Research_current_highlights",
    "List professors involved in multimodal learning.": "Professors",
    "Where is AI applied to environmental monitoring?": "Research",
    "Who leads research on real-time data processing?": "Professors",
    "Are there labs focused on mobile and embedded AI?": "Labs",
    "Which institutes research tech in civic engagement?": "Institutes",
    "What are the most exciting updates in robotics research?": "Research_current_highlights",
    "Any work being done on digital twins at Northeastern?": "Research",
    "Which professors focus on trustworthy AI?": "Professors",
    "Show me labs working on sensor-driven systems.": "Labs",
    "List institutes exploring digital governance.": "Institutes",
    "Tell me about research on adversarial machine learning.": "Research",
    "Who are the experts on edge computing?": "Professors",
    "Any collaborative research between Khoury and medical schools?": "Research",
    "Which labs are exploring swarm robotics?": "Labs",
    "What centers focus on human-AI collaboration?": "Institutes"

}

df = pd.read_csv("./intent_train_final.csv")
# df = df[(df["label"] == "Research") | (df["label"] == "Professors") | (df["label"] == "Labs")]
query_intent_map = dict()
for _ , row in df.iterrows():
    query_intent_map[row["text"]] = row["label"]

# Map each intent to a dataframe and relevant column
intent_sources = {
    "Labs": (labs, "Summary"),
    "Professors": (profs, "Research Interests"),
    "Research": (research, "Publication Summary"),
    "Institutes": (institutes, "description"),
    "Research_current_highlights": (highlights, "description"),
}

# Convert similarity score to relevance level
def score_to_relevance(score):
    if score >= 0.45:
        return 3
    elif score >= 0.25:
        return 2
    elif score >= 0.10:
        return 1
    else:
        return 0

# Build the relevance-labeled test collection
test_collection = []

for query, intent in tqdm(query_intent_map.items(), desc="Processing queries"):
    df1, text_field = intent_sources[intent]
    # Pair each document with the query
    # texts = df[text_field].fillna("").astype(str).tolist()
    # pairs = [(query, text) for text in texts]
    pairs = []
    for _, row in df1.iterrows():
        combined_text = " ".join(str(value) for value in row if pd.notna(value))
        pairs.append((query, combined_text))
        

    scores = model.predict(pairs)
    relevance_dict = {}
    for i, score in enumerate(scores):
        rel = score_to_relevance(score)
        if rel > 0:
            doc_id = str(df1.iloc[i].get("doc_id", f"doc_{intent}_{i}"))
            relevance_dict[doc_id] = rel

    test_collection.append({
        "query": query,
        "relevant_docs": relevance_dict
    })

# Save result
output_dir = "evaluations/results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "crossencoder_ndcg_test_collection.json")

with open(output_file, "w") as f:
    json.dump(test_collection, f, indent=2)

print(f"\nTest collection saved to: {output_file}")
