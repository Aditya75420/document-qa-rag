import sys
from embeddings import load_model
from vector_store import get_collection
from retrieval import retrieve
from evaluate import normalize

question, phrase = sys.argv[1], sys.argv[2]
model = load_model()
collection = get_collection()

for rank, h in enumerate(retrieve(question, model, collection, top_k=10), start=1):
    mark = "<-- ANSWER" if normalize(phrase) in normalize(h["text"]) else ""
    print(f"{rank:>2}. page {h['page']:>2}  score {h['score']:.2f}  {mark}")
    print("    ", " ".join(h["text"].split())[:100])