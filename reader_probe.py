import sys
import time

import torch
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from reader import SpanReader

TOP_K = 5
questions = sys.argv[1:] or [
    "Who painted the Mona Lisa?",
    "What is the tallest mountain in the world?",
    "Who was the first president of the United States?",
    "Why do we have seasons?",
    "What is the capital of Australia?",
]

chunks = pd.read_parquet("data/chunks.parquet")
index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", 64)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

device = "mps" if torch.backends.mps.is_available() else "cpu"
reader = SpanReader(device=device)
print("reader device:", device)
reader.read("warm up", ["warm up text"])

for question in questions:
    t0 = time.perf_counter()
    vector = encoder.encode([question], normalize_embeddings=True)
    _, ids = index.search(vector, TOP_K)
    t1 = time.perf_counter()
    rows = chunks.iloc[ids[0]]
    results = reader.read(question, rows["text"].tolist())
    t2 = time.perf_counter()

    print(f"\nQ: {question}   (retrieve {1000 * (t1 - t0):.0f} ms, read {1000 * (t2 - t1):.0f} ms)")
    best = None
    for (_, row), r in zip(rows.iterrows(), results):
        found = r["answer"] and r["score"] > r["null_score"]
        shown = r["answer"] if found else "(no answer)"
        print(f"  span {r['score']:.3f}  null {r['null_score']:.3f}  {shown[:60]!r:<64} from {row['title']}")
        if found and (best is None or r["score"] > best[0]):
            best = (r["score"], r["answer"], row["title"])
    print("  BEST:", f"{best[1]!r} from {best[2]} (score {best[0]:.3f})" if best else "no answer found")