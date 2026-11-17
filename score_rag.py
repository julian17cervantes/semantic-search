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

import rerank_settings as rs
from rag import OllamaClient, Rag
from rag_metrics import answer_correct, citation_valid, is_grounded
from searcher import Searcher

SIZE_B = {"gemma3:4b": 4, "llama3.1:8b": 8}
EF_SEARCH = 64
SOURCES_PER_QUESTION = 5

parser = argparse.ArgumentParser()
parser.add_argument("--answerable", required=True)
parser.add_argument("--unanswerable", required=True)
parser.add_argument("--models", nargs="+", required=True)
parser.add_argument("--save")
args = parser.parse_args()

chunks = pd.read_parquet("data/chunks.parquet")
index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "mps" if torch.backends.mps.is_available() else "cpu"
reranker = CrossEncoder(rs.RERANK_MODEL, device=device)
searcher = Searcher(encoder, index, chunks, reranker=reranker)
searcher.retrieve_hits("warm up", k=SOURCES_PER_QUESTION, rerank=True)

answerable = pd.read_csv(args.answerable)
unanswerable = pd.read_csv(args.unanswerable)
qa = answerable["question"].tolist()
golds = [a.split("|") for a in answerable["answer"]]
qu = unanswerable["question"].tolist()
everything = qa + qu
assert len(set(everything)) == len(everything), "duplicate questions across the two files"

hits_by_q, retrieve_ms = {}, {}
for q in everything:
    start = time.perf_counter()
    hits_by_q[q] = searcher.retrieve_hits(q, k=SOURCES_PER_QUESTION, rerank=True)
    retrieve_ms[q] = (time.perf_counter() - start) * 1000


def p95(values):
    return sorted(values)[int(len(values) * 0.95) - 1]


all_results, summary = {}, {}
for model in args.models:
    rag = Rag(lambda q: hits_by_q[q], OllamaClient(model))
    rag.answer(everything[0])
    rows = []
    for i, q in enumerate(everything):
        result = rag.answer(q)
        row = {
            "question": q,
            "answerable": i < len(qa),
            "abstained": result["abstained"],
            "answer": result["answer"],
            "raw": result["raw"],
            "citations": result["citations"],
            "invalid_citations": result["invalid_citations"],
            "citation_valid": citation_valid(result),
            "total_ms": retrieve_ms[q] + result["llm_ms"],
            "llm_ms": result["llm_ms"],
            "prompt_tokens": result["llm"].get("prompt_tokens"),
            "output_tokens": result["llm"].get("output_tokens"),
        }
        if i < len(qa):
            row["gold"] = golds[i]
            row["correct"] = answer_correct(result, golds[i])
            row["grounded_correct"] = row["correct"] and is_grounded(result, hits_by_q[q], golds[i])
        rows.append(row)
    all_results[model] = rows

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["abstained"]]
    summary[model] = {
        "answerable": len(ans),
        "correct": sum(r["correct"] for r in ans),
        "grounded_correct": sum(r["grounded_correct"] for r in ans),
        "wrongly_abstained": sum(r["abstained"] for r in ans),
        "unanswerable": len(unans),
        "unans_abstained": sum(r["abstained"] for r in unans),
        "citation_valid": sum(r["citation_valid"] for r in answered),
        "non_abstained": len(answered),
        "median_ms": statistics.median(r["total_ms"] for r in rows),
        "p95_ms": p95([r["total_ms"] for r in rows]),
    }

print(f"retrieval for all {len(everything)} questions: reranked top {rs.RERANK_DEPTH}, {SOURCES_PER_QUESTION} sources each")
print(f"\n{'model':>14} {'correct':>8} {'grounded':>9} {'wrong abst':>11} {'abst on unans':>14} {'valid cites':>12} {'median ms':>10} {'p95 ms':>8}")
for model, s in summary.items():
    print(
        f"{model:>14} {s['correct']:>5}/{s['answerable']:<2} {s['grounded_correct']:>6}/{s['answerable']:<2} "
        f"{s['wrongly_abstained']:>11} {s['unans_abstained']:>10}/{s['unanswerable']:<3} "
        f"{s['citation_valid']:>8}/{s['non_abstained']:<3} {s['median_ms']:>10.0f} {s['p95_ms']:>8.0f}"
    )

if len(args.models) > 1:
    chosen = min(
        args.models,
        key=lambda m: (-summary[m]["grounded_correct"], -summary[m]["unans_abstained"], SIZE_B.get(m, 99)),
    )
    print(f"\nSELECTED BY THE PRE-SET RULE (most grounded-correct, then more abstentions, then the smaller model): {chosen}")

for model, rows in all_results.items():
    print(f"\n===== {model}: answerable questions =====")
    for r in rows:
        if r["answerable"]:
            mark = "ok   " if r["grounded_correct"] else ("ungrounded" if r["correct"] else "WRONG")
            shown = "ABSTAINED" if r["abstained"] else r["answer"][:150].replace("\n", " ")
            print(f"  {mark:<10} gold {r['gold']}  ->  {shown}")
    print(f"===== {model}: unanswerable questions =====")
    for r in rows:
        if not r["answerable"]:
            shown = "abstained" if r["abstained"] else "ANSWERED: " + r["answer"][:150].replace("\n", " ")
            print(f"  {shown}  <-  {r['question']}")

if args.save:
    Path(args.save).parent.mkdir(exist_ok=True)
    Path(args.save).write_text(json.dumps({"summary": summary, "results": all_results}, indent=2, default=str))
    print("\nsaved", args.save)