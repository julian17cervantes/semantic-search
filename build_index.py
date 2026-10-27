import glob
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200

chunks = pd.read_parquet("data/chunks.parquet")
shards = sorted(s for s in glob.glob("data/embeddings/shard_*.npy") if ".tmp." not in s)
vectors = np.vstack([np.load(s) for s in shards])

print("chunks:", len(chunks), "vectors:", vectors.shape)
assert len(chunks) == len(vectors), "vectors and chunks are out of sync, re-run embed.py"

Path("indexes").mkdir(exist_ok=True)

flat = faiss.IndexFlatIP(vectors.shape[1])
flat.add(vectors)
faiss.write_index(flat, "indexes/flat.faiss")
print("saved exact index with", flat.ntotal, "vectors")

hnsw = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
start = time.time()
hnsw.add(vectors)
faiss.write_index(hnsw, "indexes/hnsw.faiss")
print(f"saved HNSW index in {time.time() - start:.0f} s")