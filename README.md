# RAG-based-Search-Engine

## Retrieval Task
The goal of this project is to develop a query expansion system for a chatbot that retrieves relevant research information from my university's website. The system will allow users to ask natural language questions about labs, professors, and research areas, returning ranked results rather than a single answer. This system will employ a Contextual Concept Expansion (Using Scraped Data) - Hybrid Approach, which is conceptually similar to a Retrieval-Augmented Generation (RAG) model. Instead of generating a response, the system will expand the query using contextually relevant terms and concepts extracted from scraped data (professor bios, research topics, lab descriptions). Using transformer-based models like BERT or RoBERTa, the system will expand queries by identifying semantically similar terms and concepts in the research domain. This approach will enhance the user’s query by making it more comprehensive, ensuring diverse, relevant results are retrieved and ranked, improving research discovery.

## Queries & Search Results Annotation
We will create a set of sample queries related to university research, such as "Who works on deep learning?" or "Which labs focus on renewable energy?", and annotate relevant results by mapping them to professors, labs, and research topics from the scraped website. Each query will be annotated with expanded terms based on our hybrid expansion model. For instance, if the query is about "deep learning," it might be expanded to include related terms like "neural networks," "computer vision," and "AI applications" based on the context provided by the scraped data. The relevance of search results will be determined by how closely they align with the expanded query, considering semantic similarity and coverage of subtopics. This will ensure that expanded queries retrieve a variety of relevant, ranked results.

## System Component Implementation
Our implementation will focus on the query expansion component using the Contextual Concept Expansion (Using Scraped Data) - Hybrid Approach. We will scrape research-related data from the university’s website, such as faculty bios, lab descriptions, research abstracts and research topics, to create a knowledge base for contextual expansion. Using transformer-based models like BERT or RoBERTa, we will generate contextual embeddings for research terms and use them to expand user queries with related, semantically relevant terms. The hybrid approach will combine scraped data with contextual understanding from pre-trained models to enhance the queries and improve the retrieval process. The focus will be on ensuring that queries retrieve multiple ranked results that are contextually relevant to the original user query. Evaluation will be conducted using Precision@K, Recall@K, and NDCG, assessing how effectively expanded queries improve retrieval accuracy and relevance.

## Simplified Milestones for Grading

These milestones ensure steady progress while making an "A" grade realistically achievable without unnecessary complexity. Each tier builds upon the previous one, keeping the focus on query expansion and retrieval while reducing overhead.

---

## Milestones for a B (Basic System)

### Scraping Research Data
- Scrape faculty profiles, lab descriptions, and research topics from the Khoury website.
- Store data in a structured format (CSV).

### Basic Query Expansion
- Implement simple keyword-based expansion using common synonyms.
- Expand queries with relevant terms from the scraped data.

### Basic Search & Retrieval
- Implement a BM25 keyword search to return multiple ranked results.
- Retrieve relevant professors, labs, and research based on user queries.

### Basic Evaluation
- Define evaluation metrics (Precision@K, Recall@K).
- Annotate queries with expected relevant results for testing.

---

## Milestones for a B+ (Improved Expansion)

### Contextual Query Expansion
- Use SBERT embeddings to find semantically similar terms in the scraped research data.
- Expand queries based on context, rather than just keywords.
- Remove low-relevance expansion terms.

### Evaluation Expansion
- Expand number of queries with annotated relevance judgments.
- Measure improvement in Precision@K and Recall@K.

---

## Milestones for an A- (Improved Ranking)

### Enhanced Ranking
- Implement hybrid retrieval (BM25 + semantic similarity search).
- Rank results using both keyword relevance and semantic similarity.

### Comprehensive Evaluation
- Expand number of annotated queries for testing.
- Introduce NDCG for better ranking evaluation.

---

## Milestones for an A (User Interface & Final Evaluation)

### Simple Web Interface or API
- Create a basic web page or API for user queries and ranked search results.

### Final Evaluation & Comparison
- Compare baseline vs. improved query expansion and show clear improvements.
