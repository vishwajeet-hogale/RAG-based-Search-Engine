"""
import json
import nltk
from nltk.tokenize import word_tokenize

# Load the unigram-expanded JSON file
with open("unigram_expanded.json", "r") as f:
    unigram_expansions = json.load(f)

def expand_query(user_query):
    
    nltk.download('punkt', quiet=True)  # Ensure tokenizer is available
    query_tokens = word_tokenize(user_query.lower())  # Tokenize input query
    expanded_terms = set(query_tokens)  # Start with original query terms

    # Expand each word using the unigram expansions
    for word in query_tokens:
        if word in unigram_expansions:  # Check if word has similar terms
            expanded_terms.update(unigram_expansions[word])

    # Ensure the original terms are retained and the expanded terms are sorted
    expanded_query = " ".join(sorted(expanded_terms))  # Return as space-separated string
    return expanded_query

# Example Usage
user_query = "research on natural language processing"
expanded_query = expand_query(user_query)

print("Original Query:", user_query)
print("Expanded Query:", expanded_query)

"""

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

# Download necessary resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the unigram-expanded JSON file
with open("unigram_expanded.json", "r") as f:
    unigram_expansions = json.load(f)

# Initialize spaCy NLP model for POS tagging and Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Function to filter irrelevant terms dynamically based on POS tagging and NER
def is_relevant_term(term):
    """
    Filters out irrelevant terms based on stopwords, POS tagging, and NER.
    
    Args:
        term (str): The word to check.
    
    Returns:
        bool: True if the term is considered relevant, False otherwise.
    """
    # Define stopwords
    stop_words = set(stopwords.words("english"))
    
    # Filter out stopwords
    if term in stop_words:
        return False
    
    # Use spaCy POS tagging to filter out function words (e.g., determiners, prepositions)
    doc = nlp(term)
    pos_tag = doc[0].pos_  # Get part of speech
    if pos_tag in ["DET", "ADP", "AUX", "CCONJ", "PART", "SCONJ", "PRON"]:
        return False
    
    # Named Entity Recognition (NER) - Avoid entities like organizations, locations, etc.
    if doc[0].ent_type_:
        return False
    
    return True

def expand_query(user_query):
    """
    Expands the user query using similar words from the unigram_expanded.json,
    while retaining the original query words and filtering out irrelevant terms dynamically.

    Args:
        user_query (str): The input query.
    
    Returns:
        str: The expanded query as a space-separated string with the original query intact.
    """
    query_tokens = word_tokenize(user_query.lower())  # Tokenize input query
    expanded_terms = set(query_tokens)  # Start with original query terms

    # Expand each word using the unigram expansions
    for word in query_tokens:
        if word in unigram_expansions:  # Check if word has similar terms
            expanded_terms.update(unigram_expansions[word])

    # Filter out irrelevant terms dynamically
    expanded_terms = {term for term in expanded_terms if is_relevant_term(term)}

    # Return the expanded query as a space-separated string
    expanded_query = " ".join(sorted(expanded_terms))
    return expanded_query

# Example Usage
user_query = "research on natural language processing"
expanded_query = expand_query(user_query)

print("Original Query:", user_query)
print("Expanded Query:", expanded_query)
