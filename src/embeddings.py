from sentence_transformers import SentenceTransformer
from chunking import chunk_pages
from load_documents import load_all_pdfs

MODEL_NAME = "all-MiniLM-L6-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def embed_texts(model, texts):
    """Convert a list of strings into a list of vectors."""
    return model.encode(texts, show_progress_bar=True)


if __name__ == "__main__":
    model = load_model()

    pages = load_all_pdfs()
    chunks = chunk_pages(pages)
    texts = [c["text"] for c in chunks]

    vectors = embed_texts(model, texts)
    print(f"\nEmbedded {len(texts)} chunks")
    print(f"Each vector has {len(vectors[0])} numbers")
    print(f"First 5 numbers of chunk 0: {vectors[0][:5]}")

    # Meaning test: which sentence is closest to the query?
    query = "How much does the course cost?"
    candidates = [
        "What is the program fee?",
        "The instructors have industry experience.",
        "The weather is sunny today.",
    ]
    q_vec = model.encode(query)
    c_vecs = model.encode(candidates)
    scores = model.similarity(q_vec, c_vecs)[0]
    print(f"\nQuery: {query}")
    for text, score in zip(candidates, scores):
        print(f"  {float(score):.2f}  {text}")