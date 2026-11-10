import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import faiss
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval_lib import (
    first_rank,
    hit_at,
    meets_adoption_rule,
    mrr,
    paired_bootstrap_diff,
    rerank_head,
)

parser = argparse.ArgumentParser()
parser.add_argument("queries")
parser.add_argument("--baseline", required=True, help="ranks saved by evaluate_retrieval.py for the same queries")
parser.add_argument("--depth", type=int, default=20, help="how many top chunks the reranker rescores")
parser.add_argument("--no-titles", action="store_true", help="give the reranker the chunk text without its title")
parser.add_argument("--save")
args = parser.parse_args()

K_CHUNKS = 50
EF_SEARCH = 64
MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

chunks = pd.read_parquet("data/chunks.parquet")
chunk_titles = chunks["title"].to_numpy()
chunk_texts = chunks["text"].to_numpy()
queries = pd.read_csv(args.queries)
questions = queries["query"].tolist()
expected = queries["expected"].str.split("|").apply(set).tolist()

index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "mps" if torch.backends.mps.is_available() else "cpu"
reranker = CrossEncoder(MODEL, device=device)
reranker.predict([("warm up", "warm up text")])


def passage(i):
    return chunk_texts[i] if args.no_titles else f"{chunk_titles[i]}. {chunk_texts[i]}"


vectors = encoder.encode(questions, normalize_embeddings=True)
semantic_ids, reranked_ids, rerank_ms = [], [], []
for vector, question in zip(vectors, questions):
    _, found = index.search(vector.reshape(1, -1), K_CHUNKS)
    ids = [int(i) for i in found[0] if i >= 0]
    pairs = [(question, passage(i)) for i in ids[: args.depth]]
    start = time.perf_counter()
    scores = [float(s) for s in reranker.predict(pairs, batch_size=32)]
    rerank_ms.append((time.perf_counter() - start) * 1000)
    semantic_ids.append(ids)
    reranked_ids.append(rerank_head(ids, scores, args.depth))

n = len(questions)
semantic_ranks = [first_rank(ids, e, chunk_titles) for ids, e in zip(semantic_ids, expected)]
rerank_ranks = [first_rank(ids, e, chunk_titles) for ids, e in zip(reranked_ids, expected)]

with open(args.baseline) as f:
    base = json.load(f)
assert base["queries"] == questions, "the baseline used different queries"
differences = sum(a != b for a, b in zip(semantic_ranks, base["ranks"]["semantic"]))
if differences:
    print(f"WARNING: {differences} semantic ranks differ from the saved baseline")

print(f"{args.queries}: {n} queries, rerank depth {args.depth}, titles {'off' if args.no_titles else 'on'}, device {device}")
ceiling = sum(any(chunk_titles[i] in e for i in ids[: args.depth]) for ids, e in zip(semantic_ids, expected))
print(f"candidate ceiling: the right article is among the top {args.depth} chunks for {ceiling} of {n} queries")
p50 = statistics.median(rerank_ms)
p95 = sorted(rerank_ms)[int(len(rerank_ms) * 0.95) - 1]
print(f"added latency per query (reranking only): median {p50:.1f} ms, p95 {p95:.1f} ms")

metrics = {
    "hit@1": lambda r: hit_at(r, 1),
    "hit@5": lambda r: hit_at(r, 5),
    "hit@10": lambda r: hit_at(r, 10),
    "MRR": mrr,
}
print(f"\n{'metric':>7} {'semantic':>9} {'+rerank':>8} {'change':>8}  95% interval of the change")
changes = {}
for name, fn in metrics.items():
    diff, low, high = paired_bootstrap_diff(semantic_ranks, rerank_ranks, fn)
    changes[name] = diff
    print(f"{name:>7} {fn(semantic_ranks):>9.3f} {fn(rerank_ranks):>8.3f} {diff:>+8.3f}  [{low:+.3f}, {high:+.3f}]")


def key(rank):
    return 99 if rank is None else rank


better = [q for q, a, b in zip(questions, semantic_ranks, rerank_ranks) if key(b) < key(a)]
worse = [q for q, a, b in zip(questions, semantic_ranks, rerank_ranks) if key(b) > key(a)]
print(f"\n{len(better)} queries ranked higher, {len(worse)} lower, {n - len(better) - len(worse)} unchanged")
for label, group in (("HIGHER", better), ("LOWER", worse)):
    for q in group:
        a = semantic_ranks[questions.index(q)]
        b = rerank_ranks[questions.index(q)]
        print(f"  {label}: {a or '-'} -> {b or '-'}  {q}")

adopt = meets_adoption_rule(changes["MRR"], changes["hit@10"], p50)
print("\nPRE-SET RULE (MRR gain of at least 0.03, hit@10 not lower, median added latency under 100 ms):",
      "MET" if adopt else "NOT MET")
print("(the rule applies to the run with titles on, depth 20)")

if args.save:
    Path(args.save).parent.mkdir(exist_ok=True)
    Path(args.save).write_text(
        json.dumps(
            {
                "queries_file": args.queries,
                "queries": questions,
                "settings": {"model": MODEL, "depth": args.depth, "titles": not args.no_titles},
                "ranks": {"semantic": semantic_ranks, "semantic+rerank": rerank_ranks},
                "rerank_ms": {"median": p50, "p95": p95},
            },
            indent=2,
        )
    )
    print("saved", args.save)