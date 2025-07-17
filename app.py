import streamlit as st
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os


FAQ_URL = "https://script.google.com/macros/s/AKfycbzkhSsb_mIrgvDNMv5eh5-aDDrDse5UeTzLpyutUUlJP07Ew2wJxnM96IT24vroZ_hH/exec"  

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  


@st.cache_data
def fetch_faq():
    res = requests.get(FAQ_URL)
    st.write("Web App Response:", res.text)  # For debugging
    return res.json()


def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


faq_data = fetch_faq()

questions = [item['question'] for item in faq_data]
answers = [item['answer'] for item in faq_data]
metadata = [{k: v for k, v in item.items() if k not in ['question', 'answer']} for item in faq_data]
embeddings = [get_embedding(q) for q in questions]


def get_best_answer(query):
    query_embed = get_embedding(query)
    similarities = cosine_similarity([query_embed], embeddings)[0]
    best_idx = int(np.argmax(similarities))
    return {
        "matched_question": questions[best_idx],
        "answer": answers[best_idx],
        "metadata": metadata[best_idx]
    }


st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 Property FAQ Chatbot")

query = st.text_input("Ask a question...")

if query:
    result = get_best_answer(query)
    st.markdown(f"**Matched Question:** {result['matched_question']}")
    st.markdown(f"**Answer:** {result['answer']}")
    st.markdown("**Additional Info:**")
    for key, value in result["metadata"].items():
        st.write(f"- **{key.capitalize()}**: {value}")
