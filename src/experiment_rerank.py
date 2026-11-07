import json
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder

from embeddings import load_model
from evaluate import QUESTIONS_FILE, find_rank
from experiment import CONFIGS, build_collection, make_chunks
from load_documents import load_all_pdfs
from retrieval import retrieve

RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20
CHUNKING_TO_TEST = ["fixed_500_100", "fixed_800_150", "lines_500"]
KS = [1, 3, 5, CANDIDATES]


def rerank(question, hits, reranker):
    """Reorder hits by how well each one answers the question."""
    scores = reranker.predict([(question, h["text"]) for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order]


def summarize_ks(ranks):
    n = len(ranks)
    out = {f"hit@{k}": sum(1 for r in ranks if r and r <= k) / n for k in KS}
    # MRR counts only the top 5, so it stays comparable with earlier tables
    out["mrr"] = sum(1 / r for r in ranks if r and r <= 5) / n
    return out


if __name__ == "__main__":
    questions = json.loads(Path(QUESTIONS_FILE).read_text(encoding="utf-8"))
    model = load_model()
    reranker = CrossEncoder(RERANKER_NAME)
    pages = load_all_pdfs()
    client = chromadb.PersistentClient(path="chroma_db")

    results = {}
    for label in CHUNKING_TO_TEST:
        chunks = make_chunks(pages, CONFIGS[label])
        collection = build_collection(client, label, chunks, model)

        vec_ranks, rr_ranks = [], []
        for item in questions:
            hits = retrieve(item["question"], model, collection, top_k=CANDIDATES)
            answer = item["answer_contains"]
            pgs = item.get("pages", [])
            vec_ranks.append(find_rank(hits, answer, pgs))
            reranked = rerank(item["question"], hits, reranker)
            rr_ranks.append(find_rank(reranked, answer, pgs))

        results[f"{label} | vector only"] = summarize_ks(vec_ranks)
        results[f"{label} | + reranker"] = summarize_ks(rr_ranks)

    print(f"\n{'setup':<34}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'hit@20':>8}{'mrr':>8}")
    for name, s in results.items():
        print(f"{name:<34}{s['hit@1']:>8.1%}{s['hit@3']:>8.1%}{s['hit@5']:>8.1%}{s['hit@20']:>8.1%}{s['mrr']:>8.3f}")

    Path("eval/results_rerank.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved to eval/results_rerank.json")