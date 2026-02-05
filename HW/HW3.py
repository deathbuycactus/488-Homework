import streamlit as st
from openai import OpenAI

# Show title and descr
st.title("My Lab 3 Question Answering Chatbot")

GPT = st.sidebar.selectbox("Which Model?",
                            ("mini", "regular"))
if GPT == "mini":
    model_choice = "gpt-4o-mini"
else:
    model_choice = "gpt-4o"

# Create GPT Client
if 'client' not in st.session_state:
    api_key = st.secrets["lab_key"]["IST488"]
    st.session_state.client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": "End first response: 'Do you want more information?' If they want more information continue asking if they want more until they so no and return to default prompt. Keep your answers simple enough such that a ten year old can understand them"
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
    client = st.session_state.client
    stream = client.chat.completions.create(
        model = model_choice,
        messages = st.session_state.messages, 
        stream = True)
    with st.chat_message("assistant"):
            response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})

