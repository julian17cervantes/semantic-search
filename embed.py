import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

SHARD_SIZE = 20000
OUT_DIR = Path("data/embeddings")
OUT_DIR.mkdir(parents=True, exist_ok=True)

chunks = pd.read_parquet("data/chunks.parquet")
texts = chunks["text"].tolist()
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
    elapsed = (time.time() - start) / 60
    print(f"shard {shard + 1}/{num_shards} saved ({elapsed:.1f} min elapsed)")

print("done")