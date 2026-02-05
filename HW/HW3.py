import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
from google import genai #import google.generativeai as genai

# Show title and descr
st.title("My HW 3 Question Answering Chatbot")

LLM = st.sidebar.selectbox("Which Model?",
                            ("ChatGPT", "Gemini"))

URL1 = URL1 = st.text_input(
    "Enter a URL",
    placeholder="https://example1.com"
)

URL2 = URL2 = st.text_input(
    "Enter a URL",
    placeholder="https://example2.com"
)

if LLM == "ChatGPT":
    model_choice = "gpt-4o-mini"
else:
    model_choice = "gemini-3-flash-preview"

# Create GPT Client
if LLM == "ChatGPT" and 'client' not in st.session_state:
    api_key = st.secrets["IST488"]
    st.session_state.client = OpenAI(api_key=api_key)
elif LLM == "Gemini" and 'client' not in st.session_state:
    api_key = st.secrets["IST488_G"]
    genai.configure(api_key=api_key)
    st.session_state.client = genai.Client(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": f"""
            You are a question-answering assistant.
            You will answer questions that pertain to {URL1} and/or {URL2}. Do not forget the contents of these websites.
            End first response: 'Do you want more information?' 
            If they want more information continue asking if they want more until they so no and return to default prompt. 
            Keep your answers simple enough such that a ten year old can understand them.
             """
        },
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None 
    
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# Conversation buffer
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt})    
    with st.chat_message("user"):
        st.markdown(prompt)
    if LLM == "ChatGPT":
        client = st.session_state.client
        stream = client.chat.completions.create(
            model = model_choice,
            messages = st.session_state.messages, 
            stream = True)
    elif LLM == "Gemini":
        client = st.session_state.client
        stream = client.chat.completions.create(
            model = model_choice,
            messages = st.session_state.messages, 
            stream = True)
    with st.chat_message("assistant"):
            response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

