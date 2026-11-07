import json
from pathlib import Path

import chromadb

from chunking import chunk_text
from embeddings import load_model
from evaluate import QUESTIONS_FILE, find_rank, summarize
from load_documents import load_all_pdfs
from retrieval import retrieve


def chunk_by_lines(text, max_chars=500, overlap_lines=1):
    """Pack whole lines into chunks up to max_chars, never cutting a line."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks, current, length = [], [], 0
    for line in lines:
        if current and length + len(line) + 1 > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            length = sum(len(l) + 1 for l in current)
        current.append(line)
        length += len(line) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


CONFIGS = {
    "fixed_500_100": lambda t: chunk_text(t, 500, 100),
    "fixed_300_50": lambda t: chunk_text(t, 300, 50),
    "fixed_800_150": lambda t: chunk_text(t, 800, 150),
    "lines_500": lambda t: chunk_by_lines(t, 500, 1),
    "lines_300": lambda t: chunk_by_lines(t, 300, 1),
}


def make_chunks(pages, chunker):
    out = []
    for page in pages:
        for i, piece in enumerate(chunker(page["text"])):
            out.append({
                "text": piece,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}",
            })
    return out


def build_collection(client, label, chunks, model):
    name = f"exp_{label}"
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts)
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=[v.tolist() for v in vectors],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return collection


def run_eval(collection, questions, model):
    rows = []
    for item in questions:
        hits = retrieve(item["question"], model, collection, top_k=5)
        rank = find_rank(hits, item["answer_contains"], item.get("pages", []))
        rows.append({"question": item["question"], "rank": rank})
    return rows


if __name__ == "__main__":
    questions = json.loads(Path(QUESTIONS_FILE).read_text(encoding="utf-8"))
    model = load_model()
    pages = load_all_pdfs()
    client = chromadb.PersistentClient(path="chroma_db")

    results = {}
    for label, chunker in CONFIGS.items():
        chunks = make_chunks(pages, chunker)
        collection = build_collection(client, label, chunks, model)
        rows = run_eval(collection, questions, model)
        summary = summarize(rows)
        summary["chunks"] = len(chunks)
        results[label] = {"summary": summary, "rows": rows}

    print(f"\n{'config':<16}{'chunks':>7}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'mrr':>8}")
    for label, r in results.items():
        s = r["summary"]
        print(f"{label:<16}{s['chunks']:>7}{s['hit@1']:>8.1%}{s['hit@3']:>8.1%}{s['hit@5']:>8.1%}{s['mrr']:>8.3f}")

    Path("eval/results_chunking.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nSaved to eval/results_chunking.json")