import pandas as pd
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


df = pd.read_csv("modules/data/professor_details.csv")

# use columns -> publication title, Research area

def preprocess(df):
    df.columns = df.columns.str.strip()
    df['corpus'] = df['Research Area'].fillna("")+' '+df['Publication Title'].fillna("")
    df["corpus"] = df["corpus"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower())
    df["tokenized_corpus"] = df["corpus"].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])
    return df

def train_word2vec(df):
    model = Word2Vec(sentences=df["tokenized_corpus"], vector_size=100, window =5, min_count = 2, workers=4)
    return model

def similarity_words(model, df, threshold=0.90, output_file="similar_words.json"):
    unique_words=set([word for sentence in df["tokenized_corpus"] for word in sentence])
    results={}

    for word in unique_words:
        try:
            similar_words = model.wv.most_similar(word, topn=10)
            filtered_words = [similar_word for similar_word, similarity in similar_words if similarity > threshold]
            if filtered_words:
                results[word] = filtered_words
        except KeyError:
            pass

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Words saved")

df=preprocess(df)
word2vec_model = train_word2vec(df)
similarity_words(word2vec_model, df)



