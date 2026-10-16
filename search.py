import sys
import time

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

TOP_K = 5

query = " ".join(sys.argv[1:]) or "how do airplanes stay in the air"

chunks = pd.read_parquet("data/chunks.parquet")
index = faiss.read_index("indexes/flat.faiss")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

start = time.time()
vector = model.encode([query], normalize_embeddings=True)
scores, ids = index.search(vector, TOP_K)
ms = (time.time() - start) * 1000

print(f"\nQuery: {query}  ({ms:.0f} ms)\n")
for score, i in zip(scores[0], ids[0]):
    row = chunks.iloc[i]
    snippet = row["text"][:200].replace("\n", " ")
    print(f"{score:.3f}  {row['title']}")
    print(f"       {snippet}...\n")