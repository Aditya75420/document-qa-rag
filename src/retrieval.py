from embeddings import load_model
from vector_store import get_collection
from sentence_transformers import CrossEncoder

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def retrieve(question, model, collection, top_k=3):
    """Return the top_k most relevant chunks for a question."""
    q_vec = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"],
            "score": 1 - dist,  # cosine distance -> similarity
        })
    return hits

def load_reranker():
    return CrossEncoder(RERANKER_NAME)


def retrieve_reranked(question, model, collection, reranker, top_k=5, candidates=20):
    """Fetch many candidates by vector search, then keep the best top_k after re-ranking."""
    hits = retrieve(question, model, collection, top_k=candidates)
    scores = reranker.predict([(question, h["text"]) for h in hits])
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits[:top_k]

if __name__ == "__main__":
    model = load_model()
    collection = get_collection()

    questions = [
        "What is the average salary hike?",
        "How many hiring partners are there?",
        "What topics does the course cover?",
    ]

    for q in questions:
        print(f"\n=== {q} ===")
        for hit in retrieve(q, model, collection):
            print(f"[{hit['score']:.2f}] {hit['source']}, page {hit['page']}")
            print(f"   {hit['text'][:150]!r}")