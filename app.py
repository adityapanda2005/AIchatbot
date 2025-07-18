import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_faq():
    with open("faq_data.json", "r") as f:
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
    return questions, answers, metadata

def main():
    st.set_page_config(page_title="Hodo Chatbot", page_icon="🤖")
    st.title("🤖 HODO FAQ Chatbot")

    model = load_model()
    questions, answers, metadata = load_faq()

    @st.cache_data
    def compute_embeddings():
        return model.encode(questions, convert_to_numpy=True)

    faq_embeddings = compute_embeddings()

    def get_best_answer(query):
        query_vec = model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_vec, faq_embeddings)[0]
        best_idx = int(np.argmax(sims))
        return {
            "matched_question": questions[best_idx],
            "answer": answers[best_idx],
            "metadata": metadata[best_idx],
            "score": float(sims[best_idx])
        }

    query = st.text_input("Hi, Ask a question:")

    if query:
        result = get_best_answer(query)
        st.markdown(f"**Matched Question:** {result['matched_question']}")
        st.markdown(f"**Answer:** {result['answer']}")
        st.markdown("**Extra Info:**")
        for k, v in result["metadata"].items():
            st.write(f"- **{k.capitalize()}**: {v}")

if __name__ == "__main__":
    main()
