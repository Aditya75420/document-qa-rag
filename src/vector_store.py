import chromadb
from embeddings import load_model, embed_texts
from chunking import chunk_pages
from load_documents import load_all_pdfs

DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_index():
    """Load PDFs, chunk, embed, and store everything in ChromaDB."""
    model = load_model()
    pages = load_all_pdfs()
    chunks = chunk_pages(pages)
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(model, texts)

    # Start from a clean collection so old chunks can't linger
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = get_collection()
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        documents=texts,
        embeddings=[v.tolist() for v in vectors],
        metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
    )
    return collection


if __name__ == "__main__":
    collection = build_index()
    print(f"\nStored {collection.count()} chunks in ChromaDB")

    # Quick peek at one stored record
    item = collection.get(limit=1, include=["documents", "metadatas"])
    print(f"ID: {item['ids'][0]}")
    print(f"Metadata: {item['metadatas'][0]}")
    print(f"Text: {item['documents'][0][:100]}...") 