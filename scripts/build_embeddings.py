"""Build embeddings for a collection of PDFs."""

from dotenv import load_dotenv
load_dotenv()                              # FIRST — before OpenAI client is created

import os
from src.extract import extract_text
from src.chunk import chunk_text
from src.embed import build_corpus

# build the full corpus: all PDFs -> chunks -> embed -> save
folder = "data/papers"
all_chunks = []
for fname in sorted(os.listdir(folder)):
    if fname.endswith(".pdf"):
        all_chunks.extend(chunk_text(extract_text(os.path.join(folder, fname))))

print(f"total chunks: {len(all_chunks)}")   
build_corpus(all_chunks)                     # embeds once, saves vectors.npy + chunks.json
print("corpus built and saved")