import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load all FAQ data and embeddings from the JSON file
@st.cache_data
def load_faq_embeddings():
    with open("faq_embedd.json", "r") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    metadata = [
        {
            "stage": item.get("stage", ""),
            "tags": ", ".join(item.get("tags", [])) if isinstance(item.get("tags"), list) else item.get("tags", ""),
            "property": item.get("property", ""),
            "category": item.get("category", "")
        }
        for item in data
    ]
    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data]
    return questions, answers, metadata, embeddings

questions, answers, metadata, embeddings = load_faq_embeddings()

# 🔍 Dummy embedding generator (replace with real model later if needed)
def embed_query(text):
    vec = np.array([ord(c) for c in text.lower() if c.isalnum()])
    vec = vec[:len(embeddings[0])]
    if len(vec) < len(embeddings[0]):
        vec = np.pad(vec, (0, len(embeddings[0]) - len(vec)))
    return vec.astype(np.float32)

# 🔁 Search for best-matching question
def get_best_answer(query):
    query_vec = embed_query(query)
    sims = cosine_similarity([query_vec], embeddings)[0]
    best_idx = int(np.argmax(sims))
    return {
        "matched_question": questions[best_idx],
        "answer": answers[best_idx],
        "metadata": metadata[best_idx],
        "score": sims[best_idx]
    }

# 🖥️ Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 Property FAQ Chatbot")

query = st.text_input("Ask a question:")

if query:
    result = get_best_answer(query)
    st.markdown(f"**Matched Question:** {result['matched_question']}")
    st.markdown(f"**Answer:** {result['answer']}")
    st.markdown("**Extra Info:**")
    for k, v in result["metadata"].items():
        st.markdown(f"- **{k.capitalize()}**: {v}")

        
  
