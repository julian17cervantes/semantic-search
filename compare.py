import time

import bm25s
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

try:
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
except ImportError:
    stemmer = None

K_CHUNKS = 50
TOP = 10
EF_SEARCH = 64

chunks = pd.read_parquet("data/chunks.parquet")
chunk_titles = chunks["title"].to_numpy()
all_titles = set(chunk_titles)

queries = pd.read_csv("eval/queries.csv")
queries["expected_set"] = queries["expected"].str.split("|").apply(set)
usable = queries[queries["expected_set"].apply(lambda s: bool(s & all_titles))]
usable = usable.reset_index(drop=True)
print(f"queries: {len(usable)} used, {len(queries) - len(usable)} dropped (title not in data)")
print("stemming:", "on" if stemmer else "off")

start = time.time()
corpus_tokens = bm25s.tokenize(chunks["text"].tolist(), stopwords="en", stemmer=stemmer)
bm25 = bm25s.BM25()
bm25.index(corpus_tokens)
print(f"BM25 indexed in {time.time() - start:.0f} s")
query_tokens = bm25s.tokenize(usable["query"].tolist(), stopwords="en", stemmer=stemmer)
bm25_ids, _ = bm25.retrieve(query_tokens, k=K_CHUNKS)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
query_vectors = model.encode(usable["query"].tolist(), normalize_embeddings=True)
_, semantic_ids = index.search(query_vectors, K_CHUNKS)


def first_rank(chunk_ids, expected):
    seen = []
    for i in chunk_ids:
        title = chunk_titles[i]
        if title not in seen:
            seen.append(title)
        if len(seen) == TOP:
            break
    for rank, title in enumerate(seen, 1):
        if title in expected:
            return rank
    return None


bm25_ranks = [first_rank(ids, e) for ids, e in zip(bm25_ids, usable["expected_set"])]
semantic_ranks = [first_rank(ids, e) for ids, e in zip(semantic_ids, usable["expected_set"])]


def summarize(name, ranks):
    n = len(ranks)

    def hit(k):
        return sum(r is not None and r <= k for r in ranks) / n

    mrr = sum(1 / r for r in ranks if r) / n
    print(f"{name:>9}: hit@1 {hit(1):.2f}  hit@5 {hit(5):.2f}  hit@10 {hit(10):.2f}  MRR {mrr:.3f}")


print()
summarize("BM25", bm25_ranks)
summarize("semantic", semantic_ranks)

print("\nrank of the right article (- means not in the top 10):")
for q, b, s in zip(usable["query"], bm25_ranks, semantic_ranks):
    print(f"  BM25 {b or '-':>2}  semantic {s or '-':>2}  {q}")