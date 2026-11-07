from pathlib import Path
from pypdf import PdfReader


def load_pdf(path):
    """Read one PDF and return a list of pages with metadata."""
    reader = PdfReader(path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:  # skip empty pages
            pages.append({
                "text": text,
                "source": Path(path).name,
                "page": page_number,
            })
    return pages


def load_all_pdfs(folder="data"):
    """Load every PDF inside the folder."""
    all_pages = []
    for pdf_file in sorted(Path(folder).glob("*.pdf")):
        all_pages.extend(load_pdf(pdf_file))
    return all_pages


if __name__ == "__main__":
    pages = load_all_pdfs()
    print(f"Loaded {len(pages)} pages")
    if pages:
        first = pages[0]
        print(f"\nFrom {first['source']}, page {first['page']}:\n")
        print(first["text"][:500])