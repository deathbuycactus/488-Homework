import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests

# Show title and descr
st.title("My HW 3 Question Answering Chatbot")
st.write("This is a question answering chatbot. It takes up to two URLs and will attempt to answer questions you ask about them. " \
"After your 4th prompt or after you respond that you do not want more information, it will summarize the interaction you had with it.")

LLM = st.sidebar.selectbox("Which Model?",
                            ("ChatGPT", "Gemini"))
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None 
    
URL1 = URL1 = st.text_input(
    "Enter a URL",
    placeholder="https://example1.com"
)

URL1_content = read_url_content(URL1)

URL2 = URL2 = st.text_input(
    "Enter a URL",
    placeholder="https://example2.com"
)

URL2_content = read_url_content(URL2)


if LLM == "ChatGPT":
    model_choice = "gpt-4o-mini"
else:
    model_choice = "gemini-2.5-pro"

# Create GPT Client
if LLM == "ChatGPT" and 'client' not in st.session_state:
    api_key = st.secrets["IST488"]
    st.session_state.client = OpenAI(api_key=api_key)
if LLM == "Gemini":
    genai.configure(api_key=st.secrets["IST488_G"])


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": f"""
            You are a question-answering assistant.
            You will answer questions that pertain to {URL1_content} and/or {URL2_content}. Do not forget the contents of these websites.
            End first response: 'Do you want more information?' 
            If they want more information continue asking if they want more until they say no, then summarize the conversation. 
            Keep your answers simple enough such that a ten year old can understand them.
            If you reach 3 user-assistant exchanges, answer the user's question and provide a summary of the conversation as part of your response.
            """
        },
        {
            "role": "assistant",
            "content": "How can I help you?"
        }
    ]


    
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
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})
    elif LLM == "Gemini":
        model = genai.GenerativeModel(model_choice)
        gemini_prompt = f"""
            You are a question-answering assistant.

            You will answer questions that pertain to {URL1_content} and/or {URL2_content}. Do not forget the contents of these websites.

            Rules:
            End first response: 'Do you want more information?' 
            If they want more information continue asking if they want more until they say no, then summarize the conversation. 
            Keep your answers simple enough such that a ten year old can understand them.
            If you reach 3 user-assistant exchanges, answer the user's question and provide a summary of the conversation as part of your response.
            
            User question:
            {prompt}
            """
        response = model.generate_content(gemini_prompt)
        assistant_text = response.text
        with st.chat_message("assistant"):
            st.write(assistant_text)

        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_text
        })



