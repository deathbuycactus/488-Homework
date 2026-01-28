import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

# Show title and description.
st.title("MY Document question answering")
st.write(
    "Upload a URL below and ask a question about it – GPT will answer! "
    "Choose in the sidebar whether the answer will be in 100 words, 2 connecting paragraphs, or in 5 bullet points."
)

HW_key = st.secrets["HW_key"]["IST488"]

# Create an OpenAI client.
client = OpenAI(api_key=HW_key)

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
URL = st.text_input(read_url_content())

add_selectbox = st.sidebar.selectbox(
    "How would you like the content of this website summarized?:",
    ("In 100 words", "In 2 connecting paragraphs", "In 5 bullet points")
)

if URL and add_selectbox:

    # Process the uploaded file and question.
    document = URL.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"Here's a URL: {URL} \n\n---\n\n {add_selectbox}",
        }
    ]
    if checkbox == True:
    # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )
    else:
        stream = client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            stream=True,
        )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)