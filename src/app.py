import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

from embeddings import load_model
from vector_store import get_collection
from retrieval import load_reranker
from generate import answer_question, check_citations

load_dotenv()

st.set_page_config(page_title="Document QA with RAG", page_icon="📄")


@st.cache_resource
def load_resources():
    """Load the heavy things once, not on every click."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Check your .env file.")
        st.stop()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),
    )
    return load_model(), get_collection(), client, load_reranker()


st.title("📄 Document QA with RAG")
st.caption("Ask a question. Answers come only from your document, with page citations.")

embed_model, collection, client, reranker = load_resources()
st.sidebar.write(f"Chunks in database: **{collection.count()}**")

question = st.text_input("Your question", placeholder="e.g. How many hiring partners are there?")

if st.button("Ask") and question.strip():
    with st.spinner("Searching the document and generating an answer..."):
        try:
            answer, hits = answer_question(question, embed_model, collection, client, reranker)
        except Exception as e:
            st.error(f"Request failed: {type(e).__name__}: {e}")
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    bad = check_citations(answer, hits)
    if bad:
        st.warning(f"Citation check: pages {bad} were cited but not retrieved. Verify manually.")

    with st.expander("Sources used (re-ranked chunks)"):
        for i, hit in enumerate(hits, start=1):
            st.markdown(
                f"**{i}. {hit['source']}, page {hit['page']}** "
                f"(re-rank score {hit.get('rerank_score', 0):.2f})"
            )
            st.text(hit["text"])