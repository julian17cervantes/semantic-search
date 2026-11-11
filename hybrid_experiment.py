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

from retrieval_lib import first_rank, hit_at, mrr, paired_bootstrap_diff, weighted_rrf

try:
    import Stemmer

    stemmer = Stemmer.Stemmer("english")
except ImportError:
    stemmer = None

GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
K_CHUNKS = 50
EF_SEARCH = 64

parser = argparse.ArgumentParser()
parser.add_argument("queries")
parser.add_argument("--baseline", required=True, help="ranks saved by evaluate_retrieval.py for the same queries")
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--grid", action="store_true", help="score every weight in the grid (use on the dev set)")
mode.add_argument("--frozen", action="store_true", help="score the weight frozen in hybrid_settings.py")
parser.add_argument("--save")
args = parser.parse_args()

chunks = pd.read_parquet("data/chunks.parquet")
chunk_titles = chunks["title"].to_numpy()
queries = pd.read_csv(args.queries)
questions = queries["query"].tolist()
expected = queries["expected"].str.split("|").apply(set).tolist()

corpus_tokens = bm25s.tokenize(chunks["text"].tolist(), stopwords="en", stemmer=stemmer)
bm25 = bm25s.BM25()
bm25.index(corpus_tokens)
query_tokens = bm25s.tokenize(questions, stopwords="en", stemmer=stemmer)
bm25_found, _ = bm25.retrieve(query_tokens, k=K_CHUNKS)
bm25_ids = [[int(i) for i in row] for row in bm25_found]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
semantic_ids = []
for vector in model.encode(questions, normalize_embeddings=True):
    _, found = index.search(vector.reshape(1, -1), K_CHUNKS)
    semantic_ids.append([int(i) for i in found[0] if i >= 0])


def ranks_for(weight):
    fused = [weighted_rrf([b, s], [weight, 1.0]) for b, s in zip(bm25_ids, semantic_ids)]
    return [first_rank(ids, e, chunk_titles) for ids, e in zip(fused, expected)]


with open(args.baseline) as f:
    base = json.load(f)
assert base["queries"] == questions, "the baseline used different queries"
for name, weight in (("semantic", 0.0), ("hybrid", 1.0)):
    if ranks_for(weight) != base["ranks"][name]:
        print(f"WARNING: weight {weight} does not reproduce the saved {name} ranks")

metrics = {"hit@1": lambda r: hit_at(r, 1), "hit@5": lambda r: hit_at(r, 5), "hit@10": lambda r: hit_at(r, 10), "MRR": mrr}
output = {"queries_file": args.queries, "queries": questions}

if args.grid:
    results = {w: ranks_for(w) for w in GRID}
    print(f"{args.queries}: {len(questions)} queries, {len(GRID)} weights tried")
    print(f"\n{'bm25 weight':>11} {'hit@1':>7} {'hit@5':>7} {'hit@10':>7} {'MRR':>7}")
    for w, ranks in results.items():
        cells = " ".join(f"{fn(ranks):>7.3f}" for fn in metrics.values())
        print(f"{w:>11} {cells}")
    best = max(GRID, key=lambda w: (round(mrr(results[w]), 10), -w))
    print(f"\nSELECTED BY THE PRE-SET RULE (highest dev MRR, ties to the smaller weight): {best}")
    if best == 0.0:
        print("Semantic search alone is best, so the test set is not scored.")
    output["grid_ranks"] = {str(w): r for w, r in results.items()}
    output["selected_weight"] = best
else:
    import hybrid_settings

    weight = hybrid_settings.BM25_WEIGHT
    semantic_ranks, hybrid_ranks = ranks_for(0.0), ranks_for(weight)
    print(f"{args.queries}: {len(questions)} queries, BM25 weight {weight} against semantic alone")
    print(f"\n{'metric':>7} {'semantic':>9} {'hybrid':>8} {'change':>8}  95% interval of the change")
    changes = {}
    for name, fn in metrics.items():
        diff, low, high = paired_bootstrap_diff(semantic_ranks, hybrid_ranks, fn)
        changes[name] = (diff, low, high)
        print(f"{name:>7} {fn(semantic_ranks):>9.3f} {fn(hybrid_ranks):>8.3f} {diff:>+8.3f}  [{low:+.3f}, {high:+.3f}]")

    def key(rank):
        return 99 if rank is None else rank

    better = sum(key(b) < key(a) for a, b in zip(semantic_ranks, hybrid_ranks))
    worse = sum(key(b) > key(a) for a, b in zip(semantic_ranks, hybrid_ranks))
    print(f"\n{better} queries ranked higher, {worse} lower, {len(questions) - better - worse} unchanged")
    mrr_diff, mrr_low, _ = changes["MRR"]
    if mrr_diff >= 0.03 and mrr_low > 0 and changes["hit@10"][0] >= 0:
        verdict = "ADOPTION RULE MET (confirmed on the held-out set)"
    elif mrr_diff > 0:
        verdict = "NOT CONFIRMED (the change is positive but its interval includes zero, or it is under 0.03)"
    else:
        verdict = "DID NOT REPLICATE (no improvement)"
    print("\nPRE-SET RULE:", verdict)
    output.update({"bm25_weight": weight, "ranks": {"semantic": semantic_ranks, "hybrid": hybrid_ranks}})

if args.save:
    Path(args.save).parent.mkdir(exist_ok=True)
    Path(args.save).write_text(json.dumps(output, indent=2))
    print("saved", args.save)