import json
from nltk.tokenize import word_tokenize

def load_similar_words(file_path="similar_words.json"):
    with open(file_path, "r") as f:
        similar_words_dict = json.load(f)
    return similar_words_dict

def simple_expansion(query, similar_words_dict):
    query_tokens = word_tokenize(query.lower())
    expanded_query = set(query_tokens)

    for word in query_tokens:
        if word in similar_words_dict:
            expanded_query.update(similar_words_dict[word])

    return " ".join(expanded_query)

similar_words_dict = load_similar_words("similar_words.json")

user_query = "People that teach deep learning"
expanded_query = simple_expansion(user_query, similar_words_dict)
print("Expanded query: ", expanded_query)