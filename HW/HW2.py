import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a URL below and ask a question about it and choose an LLM will answer! "
    "Choose in the sidebar what LLM to use, and whether the answer will be in 100 words, 2 connecting paragraphs, or in 5 bullet points."
)

if st.checkbox("Advanced Model"):
    st.write("Advanced Model: ON")
    checkbox = True
else:
    st.write("Advanced Model: OFF")
    checkbox = False

# URL read function

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None    
    
# Let the user enter a URL
URL = URL = st.text_input(
    "Enter a URL",
    placeholder="https://example.com"
)

summary = st.sidebar.selectbox(
    "How would you like the content of this website summarized?:",
    ("In 100 words", "In 2 connecting paragraphs", "In 5 bullet points")
)

LLM = st.sidebar.selectbox(
    "Which LLM would you like to use?",
    ("ChatGPT", "Gemini")
)

output_lang = st.selectbox(
    "What language would you like the response to be in?",
    ("English", "French", "Spanish")
)
# Api keys
#st.write("Secrets keys:", list(st.secrets.keys()))
HW_key = st.secrets["IST488"] # OpenAI Key
Gemini_Key = st.secrets['IST488_G'] # Gemini Key

if URL and summary:

    # Process the URL and question.
    content = read_url_content(URL)
    messages = [
        {
            "role": "user",
            "content": f"Here's a some content of a webpage: {content} \n\n---\n\n Respond in {output_lang} {summary}",
        }
    ]
    if LLM == "ChatGPT":
        client = OpenAI(api_key=HW_key)

        model = "gpt-5-nano" if checkbox else "gpt-5-mini"

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        st.write_stream(stream)

    elif LLM == "Gemini":
        genai.configure(api_key=Gemini_Key)

        model = (
            "gemini-2.5-flash" if checkbox else "gemini-2.5-pro"
        )

        model = genai.GenerativeModel(model)

        response = model.generate_content(messages[0]["content"])

        st.write(response.text)