import pandas as pd
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')



df = pd.read_csv("modules/data/professor_details.csv")

print(df.columns)

# use columns -> publication title, Research area

def preprocess(df):
    df.columns = df.columns.str.strip()
    df['corpus'] = df['Research Area'].fillna("")+' '+df['Publication Title'].fillna("")
    df["corpus"] = df["corpus"].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).lower())
    df["tokenized_corpus"] = df["corpus"].apply(word_tokenize)
    return df

def train_word2vec(df):
    model = Word2Vec(sentences=df["tokenized_corpus"], vector_size=100, window =5, min_count = 2, workers=4)
    return model

def similarity_words(model, df, threshold=0.90, output_file="similar_words.txt"):
    unique_words=set([word for sentence in df["tokenized_corpus"] for word in sentence])
    results=[]

    for word in unique_words:
        try:
            similar_words = model.wv.most_similar(word, topn=10)
            for similar_word, similarity in similar_words:
                if similarity>threshold:
                    #results.append(f"{word} - {similar_word}\n")
                    results.append(f"{word}: {', '.join(map(str,similar_words))}\n")
        except KeyError:
            pass

    with open(output_file, "w") as f:
        f.writelines(results)

    print(f"Words saved")

df=preprocess(df)
word2vec_model = train_word2vec(df)
similarity_words(word2vec_model, df)



