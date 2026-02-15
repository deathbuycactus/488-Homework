import sys
from PyPDF2 import PdfReader
from pathlib import Path

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
        api_key=st.secrets["lab_key"]["IST488"]
    )

# ==============================
# Initialize ChromaDB
# ==============================
chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_Lab')
collection = chroma_client.get_or_create_collection('Lab4Collection')

# ==============================
# PDF Embedding Functions
# ==============================
def add_to_collection(collection, text, file_name):
    """Embed a PDF document and store in ChromaDB."""
    client = st.session_state.openai_client
    query_embed = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    embedding = query_embed.data[0].embedding
    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{"source": file_name}]
    )

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text is not None:
            text += page_text
    return text

def load_pdfs_to_collection(folder_path, collection):
    folder = Path(folder_path)
    for html_path in folder.glob("*.html"):
        try:
            text = extract_text_from_pdf(html_path)
            file_name = html_path.name
            add_to_collection(collection, text, file_name)
        except Exception as e:
            print(f"Error processing {html_path.name}: {e}")

# Load PDFs if collection is empty
if collection.count() == 0:
    load_pdfs_to_collection('./SYLLABI/', collection)

# ==============================
# Streamlit UI
# ==============================
st.title('Lab 4: Chatbot using RAG')
st.write("This chatbot answers questions using your PDFs. External sources may be used if needed.")

LLM = st.sidebar.selectbox("Which Model?", ("ChatGPT",))
model_choice = "gpt-4o-mini" if LLM == "ChatGPT" else None

# ==============================
# Session State
# ==============================
if "Lab4_VectorDB" not in st.session_state:
    st.session_state.Lab4_VectorDB = collection

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": (
                "You are a question-answering assistant. "
                "If the question cannot be answered using the PDF texts given, "
                "you may use external sources. "
                "Cite sources if used."
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

    client = st.session_state.openai_client

    # Step 1: Embed user question
    query_embed = client.embeddings.create(
        input=prompt,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Step 2: Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=3
    )

    # Step 3: Combine retrieved documents
    retrieved_text = "\n".join(results["documents"][0])

    # Step 4: Inject context into messages
    context_message = {
        "role": "system",
        "content": f"Use this retrieved context to answer the user:\n{retrieved_text}"
    }
    messages_with_context = st.session_state.messages + [context_message]

    # Step 5: Call GPT
    stream = client.chat.completions.create(
        model=model_choice,
        messages=messages_with_context,
        stream=True
    )

    # Step 6: Display GPT response
    with st.chat_message("assistant"):
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})