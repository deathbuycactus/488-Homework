import sys
import pysqlite3 as sqlite3

# ChromaDB fix (same as Streamlit app)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import streamlit as st
import pandas as pd
from openai import OpenAI

# Use your API key (env variable recommended)
client = OpenAI(api_key=st.secrets["IST488"])

# SAME path as Streamlit app
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')

# Create collection if it doesn't exist
collection = chroma_client.get_or_create_collection('HW7_Collection')


def add_to_collection(text, doc_id):
    embedding = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    ).data[0].embedding

    collection.add(
        documents=[text],
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"source": doc_id}]
    )


def load_csv_to_chroma(csv_path):
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))

        full_text = f"{title}\n{content}"

        add_to_collection(full_text, f"doc_{idx}")


if __name__ == "__main__":
    load_csv_to_chroma("ChromaDB_for_HW/news.csv")
    print("✅ Vector DB built successfully.")