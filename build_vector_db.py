import sys
import pysqlite3 as sqlite3

# ChromaDB fix (same as Streamlit app)
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import streamlit as st
import pandas as pd
from openai import OpenAI

# OpenAI client from environment variable
api_key = st.secrets["IST488"]
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=api_key)

chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')
collection = chroma_client.get_or_create_collection('HW7_Collection')

def add_batch_to_collection(texts, ids):
    embeddings = client.embeddings.create(input=texts, model='text-embedding-3-small')
    embeddings_list = [e.embedding for e in embeddings.data]

    collection.add(
        documents=texts,
        ids=ids,
        embeddings=embeddings_list,
        metadatas=[{"source": i} for i in ids]
    )

def load_csv_to_chroma(csv_path, batch_size=10):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    texts = []
    ids = []
    for idx, row in df.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        full_text = f"{title}\n{content}"

        texts.append(full_text)
        ids.append(f"doc_{idx}")

        if len(texts) == batch_size:
            add_batch_to_collection(texts, ids)
            print(f"Processed {idx+1} / {len(df)} rows")
            texts, ids = [], []

    # Add remaining
    if texts:
        add_batch_to_collection(texts, ids)

    print("✅ Vector DB build complete.")

if __name__ == "__main__":
    load_csv_to_chroma("./ChromaDB_for_HW/news.csv", batch_size=20)