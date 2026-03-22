# build_vector_db.py
from pathlib import Path
import pandas as pd
import sys
import pysqlite3  # SQLite fix for ChromaDB

# Patch sqlite3 before importing chromadb
sys.modules['sqlite3'] = pysqlite3
import chromadb
from openai import OpenAI
import streamlit as st  # Only for st.secrets
import os
# ===========================
# OpenAI client
# ===========================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)

# ===========================
# ChromaDB
# ===========================
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')
collection = chroma_client.get_or_create_collection('HW7_Collection')

# ===========================
# Helper functions
# ===========================
def add_batch_to_collection(texts, ids, dates):
    # Embed a batch of documents
    embeddings_resp = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings_list = [e.embedding for e in embeddings_resp.data]

    collection.add(
    documents=texts,
    ids=ids,
    embeddings=embeddings_list,
    metadatas=[{"source": i, "date": d} for i, d in zip(ids, dates)]
)

def load_csv_to_chroma(csv_path, batch_size=10):
    csv_file = Path(csv_path).resolve()
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")

    texts, ids, dates = [], [], []
    for idx, row in df.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        date = str(row.get("date", ""))
        
        full_text = f"{title}\n{content}"
        
        dates.append(date)
        texts.append(full_text)
        ids.append(f"doc_{idx}")

        if len(texts) >= batch_size:
            add_batch_to_collection(texts, ids, dates)
            texts, ids, dates = [], [], []

    # Add remaining docs
    if texts:
        add_batch_to_collection(texts, ids, dates)

    print("✅ Vector DB build complete.")
    print(f"Total documents in collection: {collection.count()}")

# ===========================
# Run builder
# ===========================
if __name__ == "__main__":
    load_csv_to_chroma("./ChromaDB_for_HW/news.csv", batch_size=20)