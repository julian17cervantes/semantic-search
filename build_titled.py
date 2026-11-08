import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from pathlib import Path

import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SHARD_SIZE = 20000
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
OUT_DIR = Path("data/embeddings_titled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

chunks = pd.read_parquet("data/chunks.parquet")
texts = (chunks["title"] + ". " + chunks["text"]).tolist()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

num_shards = (len(texts) + SHARD_SIZE - 1) // SHARD_SIZE
start = time.time()
for shard in range(num_shards):
    path = OUT_DIR / f"shard_{shard:04d}.npy"
    if path.exists():
        print(f"shard {shard + 1}/{num_shards}: already done, skipping")
        continue
    batch = texts[shard * SHARD_SIZE : (shard + 1) * SHARD_SIZE]
    vectors = model.encode(batch, batch_size=64, normalize_embeddings=True)
    tmp = OUT_DIR / f"shard_{shard:04d}.tmp.npy"
    np.save(tmp, vectors.astype(np.float32))
    tmp.rename(path)
    print(f"shard {shard + 1}/{num_shards} saved ({(time.time() - start) / 60:.1f} min elapsed)")

shards = sorted(p for p in OUT_DIR.glob("shard_*.npy") if ".tmp." not in p.name)
vectors = np.vstack([np.load(p) for p in shards])
assert len(vectors) == len(chunks), "vectors and chunks are out of sync"

hnsw = faiss.IndexHNSWFlat(vectors.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
hnsw.add(vectors)
Path("indexes").mkdir(exist_ok=True)
faiss.write_index(hnsw, "indexes/hnsw_titled.faiss")
print("saved indexes/hnsw_titled.faiss with", hnsw.ntotal, "vectors")