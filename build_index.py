import glob
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

chunks = pd.read_parquet("data/chunks.parquet")
shards = sorted(glob.glob("data/embeddings/shard_*.npy"))
shards = [s for s in shards if ".tmp." not in s]
vectors = np.vstack([np.load(s) for s in shards])

print("chunks:", len(chunks), "vectors:", vectors.shape)
assert len(chunks) == len(vectors), "vectors and chunks are out of sync, re-run embed.py"

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

Path("indexes").mkdir(exist_ok=True)
faiss.write_index(index, "indexes/flat.faiss")
print("saved index with", index.ntotal, "vectors")