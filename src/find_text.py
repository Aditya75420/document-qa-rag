import sys
from load_documents import load_all_pdfs

phrase = " ".join(sys.argv[1:]).lower()
for p in load_all_pdfs():
    if phrase in " ".join(p["text"].split()).lower():
        print(f"{p['source']}  page {p['page']}")