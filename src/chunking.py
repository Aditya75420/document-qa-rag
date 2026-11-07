from load_documents import load_all_pdfs


def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks of roughly chunk_size characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def chunk_pages(pages, chunk_size=800, overlap=150):
    """Chunk every page, keeping source and page number on each chunk."""
    all_chunks = []
    for page in pages:
        pieces = chunk_text(page["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}",
            })
    return all_chunks


if __name__ == "__main__":
    pages = load_all_pdfs()
    chunks = chunk_pages(pages)
    print(f"{len(pages)} pages became {len(chunks)} chunks")

    # find two consecutive chunks from the same page
    for a, b in zip(chunks, chunks[1:]):
        if a["page"] == b["page"]:
            print(f"\nChunk A ({a['chunk_id']}), last 150 characters:\n")
            print(a["text"][-150:])
            print(f"\nChunk B ({b['chunk_id']}), first 150 characters:\n")
            print(b["text"][:150])
            break