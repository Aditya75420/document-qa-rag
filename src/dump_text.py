# src/dump_text.py
from load_documents import load_all_pdfs

with open("eval/extracted_text.txt", "w", encoding="utf-8") as f:
    for p in load_all_pdfs():
        f.write(f"\n=== page {p['page']} ===\n{p['text']}\n")