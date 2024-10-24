from ast import If
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Model:
    vectorizer: TfidfVectorizer
    df: pd.DataFrame

    def __init__(self):
        # Get tf-idf matrix using fit_transform function
        self.vectorizer = TfidfVectorizer()
        self.df = pd.read_csv("dataset/train.csv")
        # Hapus baris dengan nilai NaN
        self.df = self.df.dropna(subset=["text"])
        # pakai 50 row pertama saja
        self.df = self.df.iloc[:50]

    def train(self):
        self.X = self.vectorizer.fit_transform(
            self.df["text"]
        )  # Store tf-idf representations of all docs

    def search(self, query: str):
        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.X, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc

        response = []
        # return results.argsort()[-3:][::-1]
        for i in results.argsort()[-3:][::-1]:
            response.append(
                {"No": i, "Dokumen": self.df.iloc[i, 0], "score": results[i]}
            )

        return pd.DataFrame(response)


model = Model()

st.title("Mesin Pencarian")

# Create a text element and let the reader know the data is loading.
train_state = st.text("Training model...")
# Load 10,000 rows of data into the dataframe.
model.train()
# Notify the reader that the data was successfully loaded.
train_state.text("Training model...done!")
model.df

with st.form("Search something.."):
    query = st.text_input("query")
    submit = st.form_submit_button("search")

if submit:
    result = model.search(query)
    result

# st.sidebar.title("Cari: ")
# query = st.sidebar.text_input("input an text")
# st.sidebar.button("query", on_click=print(query))
