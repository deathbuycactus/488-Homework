import sys
from PyPDF2 import PdfReader
from pathlib import Path
import pysqlite3 as sqlite3

# ChromaDB fix
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from openai import OpenAI
import chromadb

# ==============================
# Initialize OpenAI client
# ==============================
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(
        api_key=st.secrets["IST488"]
    )

# ==============================
# Load pre-built ChromaDB
# ==============================
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW')

try:
    collection = chroma_client.get_collection('HW7_Collection')
except chromadb.errors.NotFoundError:
    st.error("❌ Collection 'HW7_Collection' not found in ChromaDB. Make sure the DB was built outside the app.")
    st.stop()

# Display number of articles for debug
st.write(f"Number of articles in DB: {collection.count()}")

# ==============================
# HTML Embedding Functions
# ==============================
def relative_news_info(query, n_results=5, call_llm=False):
    """
    Searches ChromaDB for relevant news info.
    If call_llm=True, returns an LLM-generated answer using retrieved context.
    Otherwise returns retrieved text only.
    """
    client = st.session_state.openai_client

    # Embed query
    embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Search vector DB
    results = st.session_state.HW7_VectorDB.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    # Handle empty results
    if not results["documents"][0]:
        return "No news articles found for your query."

    retrieved_text = "\n".join(results["documents"][0])

    if not call_llm:
        return retrieved_text

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using only this context:\n" + retrieved_text},
            {"role": "user", "content": query}
        ]
    )

    return response.choices[0].message.content

# ==============================
# Streamlit UI
# ==============================
st.title('HW 7: News Information Bot')
st.write("This chatbot answers news related questions using a collection of articles in a CSV file (pre-loaded into ChromaDB).")

LLM = st.sidebar.selectbox("Which Model?", ("ChatGPT",))
model_choice = "gpt-4o-mini" if LLM == "ChatGPT" else None

# ==============================
# Session State
# ==============================
if "HW7_VectorDB" not in st.session_state:
    st.session_state.HW7_VectorDB = collection

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": (
                "You are a question-answering assistant. "
                "If the question cannot be answered using the content of the CSV file given, do not use external sources to answer and instead state your uncertainty."
            )
        },
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# ==============================
# Chat input + RAG retrieval
# ==============================
if prompt := st.chat_input("Ask a question:"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context from ChromaDB
    retrieved_text = relative_news_info(prompt)

    # Inject context into messages
    context_message = {
        "role": "system",
        "content": f"Use this retrieved context to answer the user:\n{retrieved_text}"
    }
    messages_with_context = st.session_state.messages + [context_message]

    # Call GPT
    stream = st.session_state.openai_client.chat.completions.create(
        model=model_choice,
        messages=messages_with_context,
        stream=True
    )

    # Display GPT response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Keep last MAX_INTERACTIONS
    MAX_INTERACTIONS = 5
    system_msg = st.session_state.messages[0]
    conversation = st.session_state.messages[1:]
    conversation = conversation[-MAX_INTERACTIONS*2:]
    st.session_state.messages = [system_msg] + conversation