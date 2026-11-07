import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from embeddings import load_model
from vector_store import get_collection
from retrieval import retrieve, retrieve_reranked, load_reranker

load_dotenv()

# Smaller model first because the bigger one kept timing out
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
TOP_K = 5


def build_prompt(question, hits):
    """Put the retrieved chunks and the rules into one prompt."""
    context = ""
    for i, hit in enumerate(hits, start=1):
        context += f"[Excerpt {i} | {hit['source']} | page {hit['page']}]\n{hit['text']}\n\n"

    return f"""You are a document question-answering assistant.
Answer the question using ONLY the excerpts below.
After each fact, cite the source like this: (file name, page X).
Only cite page numbers that appear in the excerpt labels above, and only for excerpts you actually used. Never guess a page number.
If the answer is not in the excerpts, reply exactly: "I don't know based on the provided document."
Do not use outside knowledge.

EXCERPTS:
{context}
QUESTION: {question}

ANSWER:"""


def check_citations(answer, hits):
    """Return cited pages that were NOT in the retrieved chunks."""
    retrieved_pages = {h["page"] for h in hits}
    cited_pages = {int(p) for p in re.findall(r"page (\d+)", answer, re.IGNORECASE)}
    return sorted(cited_pages - retrieved_pages)


def answer_question(question, model, collection, client, reranker=None):
    if reranker is not None:
        hits = retrieve_reranked(question, model, collection, reranker, top_k=TOP_K)
    else:
        hits = retrieve(question, model, collection, top_k=TOP_K)
    prompt = build_prompt(question, hits)

    last_error = None
    for gemini_model in GEMINI_MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                )
                print(f"  [answered by {gemini_model}]")
                return response.text, hits
            except Exception as e:
                last_error = e
                wait = 2 ** attempt * 3  # 3s, 6s, 12s
                print(f"  ({gemini_model} failed: {str(e)[:150]}, retrying in {wait}s)")
                time.sleep(wait)
    raise last_error


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not found. Check your .env file.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),
    )
    embed_model = load_model()
    collection = get_collection()
    reranker = load_reranker()

    questions = [
        "What is the average salary hike?",
        "How many hiring partners are there?",
        "What is the capital of France?",  # not in the document on purpose
    ]

    for q in questions:
        print(f"\nQ: {q}")
        try:
            answer, hits = answer_question(q, embed_model, collection, client, reranker)
            print(f"A: {answer}")
            print("  Retrieved (page, score):", [(h["page"], round(h["score"], 2)) for h in hits])
            bad = check_citations(answer, hits)
            if bad:
                print(f"  WARNING: cited pages not in retrieved chunks: {bad}")
        except Exception as e:
            print(f"Request failed: {type(e).__name__}: {e}")