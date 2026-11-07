import json
import re
import sys
from pathlib import Path

from embeddings import load_model
from vector_store import get_collection
from retrieval import retrieve, retrieve_reranked, load_reranker

QUESTIONS_FILE = Path("eval/questions.json")
K_VALUES = [1, 3, 5]


def normalize(text):
    return re.sub(r"\s+", " ", text).lower().strip()


def find_rank(hits, answer_contains, pages):
    """1-based rank of the first correct chunk, or None if not retrieved."""
    target = normalize(answer_contains)
    for rank, hit in enumerate(hits, start=1):
        has_text = target in normalize(hit["text"])
        right_page = (not pages) or hit["page"] in pages
        if has_text and right_page:
            return rank
    return None


def summarize(rows):
    n = len(rows)
    summary = {}
    for k in K_VALUES:
        found = sum(1 for r in rows if r["rank"] and r["rank"] <= k)
        summary[f"hit@{k}"] = round(found / n, 3)
    summary["mrr"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)
    return summary


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = args[0] if args else "baseline"
    use_rerank = "--rerank" in sys.argv

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    model = load_model()
    collection = get_collection()
    reranker = load_reranker() if use_rerank else None

    rows = []
    for item in questions:
        if use_rerank:
            hits = retrieve_reranked(item["question"], model, collection, reranker, top_k=max(K_VALUES))
        else:
            hits = retrieve(item["question"], model, collection, top_k=max(K_VALUES))
        rank = find_rank(hits, item["answer_contains"], item.get("pages", []))
        rows.append({"question": item["question"], "rank": rank})

    mode = "vector + reranker" if use_rerank else "vector only"
    print(f"\n=== Results: {label} ({mode}, {len(rows)} questions) ===")
    for r in rows:
        status = f"rank {r['rank']}" if r["rank"] else "MISS"
        print(f"  [{status:>7}]  {r['question']}")

    summary = summarize(rows)
    print("\nSummary:")
    for name, value in summary.items():
        print(f"  {name}: {value:.1%}" if name != "mrr" else f"  {name}: {value:.3f}")

    out = Path("eval") / f"results_{label}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")