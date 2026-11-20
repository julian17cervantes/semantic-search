import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

import torch
import faiss
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

import rerank_settings as rs
from rag_metrics import contains_gold
from searcher import Searcher

question, golds = sys.argv[1], sys.argv[2].split("|")
chunks = pd.read_parquet("data/chunks.parquet")
index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", 64)
device = "mps" if torch.backends.mps.is_available() else "cpu"
searcher = Searcher(
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    index,
    chunks,
    reranker=CrossEncoder(rs.RERANK_MODEL, device=device),
)
print("Q:", question, "  gold:", golds)
for i, hit in enumerate(searcher.retrieve_hits(question, k=5, rerank=True), 1):
    found = contains_gold(f"{hit['title']}. {hit['text']}", golds)
    print(f"[{i}] contains gold: {'YES' if found else 'no '}  {hit['title']}: {hit['text'][:110]!r}")