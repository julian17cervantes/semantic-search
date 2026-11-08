import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import torch 
import bm25s
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from retrieval_lib import bootstrap_ci, first_rank, hit_at, mrr, rrf

try:
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
except ImportError:
    stemmer = None

parser = argparse.ArgumentParser()
parser.add_argument("queries", help="CSV with columns query and expected (alternatives separated by |)")
parser.add_argument("--check-only", action="store_true", help="only check that the expected titles exist")
parser.add_argument("--details", action="store_true", help="print the rank of every query")
parser.add_argument("--save", help="write per-query ranks to this JSON file")
args = parser.parse_args()

K_CHUNKS = 50
EF_SEARCH = 64

chunks = pd.read_parquet("data/chunks.parquet")
chunk_titles = chunks["title"].to_numpy()
all_titles = set(chunk_titles)

queries = pd.read_csv(args.queries)
queries["expected_set"] = queries["expected"].str.split("|").apply(set)
missing = queries[~queries["expected_set"].apply(lambda s: bool(s & all_titles))]
if len(missing):
    print("No expected title exists in the corpus for these queries. Fix them first:")
    for _, row in missing.iterrows():
        word = row["expected"].split("|")[0].lower()
        close = sorted(t for t in all_titles if word in t.lower())[:5]
        print(f"  {row['query']} -> {row['expected']}")
        print(f"      similar titles: {close}")
    raise SystemExit(1)
print(f"{args.queries}: {len(queries)} queries, all expected titles exist")
if args.check_only:
    raise SystemExit(0)
print("stemming:", "on" if stemmer else "off")

corpus_tokens = bm25s.tokenize(chunks["text"].tolist(), stopwords="en", stemmer=stemmer)
bm25 = bm25s.BM25()
bm25.index(corpus_tokens)
query_tokens = bm25s.tokenize(queries["query"].tolist(), stopwords="en", stemmer=stemmer)
bm25_ids, _ = bm25.retrieve(query_tokens, k=K_CHUNKS)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
vectors = model.encode(queries["query"].tolist(), normalize_embeddings=True)
semantic_ids = []
for vector in vectors:
    _, found = index.search(vector.reshape(1, -1), K_CHUNKS)
    semantic_ids.append(found[0])

hybrid_ids = [rrf([b, s]) for b, s in zip(bm25_ids, semantic_ids)]
expected = queries["expected_set"]
methods = {
    "BM25": [first_rank(ids, e, chunk_titles) for ids, e in zip(bm25_ids, expected)],
    "semantic": [first_rank(ids, e, chunk_titles) for ids, e in zip(semantic_ids, expected)],
    "hybrid": [first_rank(ids, e, chunk_titles) for ids, e in zip(hybrid_ids, expected)],
}

metrics = {
    "hit@1": lambda r: hit_at(r, 1),
    "hit@5": lambda r: hit_at(r, 5),
    "hit@10": lambda r: hit_at(r, 10),
    "MRR": mrr,
}
print(f"\n{len(queries)} queries, with 95% bootstrap intervals in brackets")
print(f"{'method':>9} " + " ".join(f"{name:>22}" for name in metrics))
for method, ranks in methods.items():
    cells = []
    for fn in metrics.values():
        low, high = bootstrap_ci(ranks, fn)
        cells.append(f"{fn(ranks):.2f} [{low:.2f}-{high:.2f}]")
    print(f"{method:>9} " + " ".join(f"{cell:>22}" for cell in cells))

if args.details:
    print("\nrank of the right article (- means not in the top 10):")
    for i, query in enumerate(queries["query"]):
        b, s, h = (methods[m][i] for m in ("BM25", "semantic", "hybrid"))
        print(f"  BM25 {b or '-':>2}  semantic {s or '-':>2}  hybrid {h or '-':>2}  {query}")

if args.save:
    Path(args.save).parent.mkdir(exist_ok=True)
    Path(args.save).write_text(
        json.dumps({"queries_file": args.queries, "queries": queries["query"].tolist(), "ranks": methods}, indent=2)
    )
    print("saved", args.save)