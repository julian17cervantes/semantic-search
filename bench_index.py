import glob
import time

import faiss
import numpy as np

K = 10
NUM_QUERIES = 1000
EF_SEARCH_VALUES = [16, 32, 64, 128]

shards = sorted(s for s in glob.glob("data/embeddings/shard_*.npy") if ".tmp." not in s)
vectors = np.vstack([np.load(s) for s in shards])
dim = vectors.shape[1]

# Use random chunks as queries. Each one finds itself first, so drop that match.
rng = np.random.default_rng(1)
query_ids = rng.choice(len(vectors), NUM_QUERIES, replace=False)
queries = vectors[query_ids]


def drop_self(ids, query_ids):
    return np.array(
        [[i for i in row if i != qid][:K] for row, qid in zip(ids, query_ids)]
    )


def ms_per_query(index, n=200):
    start = time.perf_counter()
    for q in queries[:n]:
        index.search(q.reshape(1, -1), K + 1)
    return (time.perf_counter() - start) / n * 1000


flat = faiss.read_index("indexes/flat.faiss")
_, exact = flat.search(queries, K + 1)
exact = drop_self(exact, query_ids)
print(f"exact (flat): {ms_per_query(flat):.2f} ms per query, recall 1.000")

hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 200
start = time.time()
hnsw.add(vectors)
print(f"HNSW build time: {time.time() - start:.0f} s")
faiss.write_index(hnsw, "indexes/hnsw.faiss")

for ef in EF_SEARCH_VALUES:
    hnsw.hnsw.efSearch = ef
    _, approx = hnsw.search(queries, K + 1)
    approx = drop_self(approx, query_ids)
    recall = np.mean([len(set(a) & set(e)) / K for a, e in zip(approx, exact)])
    print(f"HNSW efSearch={ef}: {ms_per_query(hnsw):.2f} ms per query, recall@{K} {recall:.3f}")