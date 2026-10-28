import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import time

import torch
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

from answer_metrics import is_correct, normalize
from reader import SpanReader

MAX_K = 10
KS = [3, 5, 10]

chunks = pd.read_parquet("data/chunks.parquet")
rows = pd.read_csv("eval/answers_dev.csv")
questions = rows["question"].tolist()
golds = [a.split("|") for a in rows["answer"]]

index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", 64)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = encoder.encode(questions, normalize_embeddings=True)
ids = []
for vector in vectors:
    _, row_ids = index.search(vector.reshape(1, -1), MAX_K)
    ids.append(row_ids[0])

device = "mps" if torch.backends.mps.is_available() else "cpu"
reader = SpanReader(device=device)
reader.read("warm up", ["warm up text"])


def contains_answer(text, gold_answers):
    padded = f" {normalize(text)} "
    return any(f" {normalize(g)} " in padded for g in gold_answers if normalize(g))


def pick_best(reads):
    candidates = [r for r in reads if r["answer"] and r["score"] > r["null_score"]]
    if not candidates:
        return None
    best = max(candidates, key=lambda r: r["score"])
    return best["answer"], best["score"]


def pick_vote(reads):
    totals, first = {}, {}
    for r in reads:
        if r["answer"] and r["score"] > r["null_score"]:
            key = normalize(r["answer"])
            if key:
                totals[key] = totals.get(key, 0.0) + r["score"]
                first.setdefault(key, r["answer"])
    if not totals:
        return None
    key = max(totals, key=totals.get)
    return first[key], totals[key]


print("CEILING: questions whose answer appears in the top-k chunks")
for k in KS:
    found = sum(
        any(contains_answer(t, g) for t in chunks.iloc[row_ids[:k]]["text"])
        for g, row_ids in zip(golds, ids)
    )
    print(f"  k={k:<2}: {found}/{len(questions)}")


def read_all(use_title):
    out, start = [], time.perf_counter()
    for q, row_ids in zip(questions, ids):
        sub = chunks.iloc[row_ids]
        texts = [f"{t}. {x}" if use_title else x for t, x in zip(sub["title"], sub["text"])]
        out.append(reader.read(q, texts))
    ms = (time.perf_counter() - start) * 1000 / len(questions)
    return out, ms


results = []
for use_title in (False, True):
    all_reads, ms = read_all(use_title)
    label = "with" if use_title else "without"
    print(f"\nreading {MAX_K} chunks {label} titles: {ms:.0f} ms per question")
    for k in KS:
        for name, picker in (("best", pick_best), ("vote", pick_vote)):
            correct = answered = 0
            misses = []
            for q, g, reads in zip(questions, golds, all_reads):
                picked = picker(reads[:k])
                if picked is None:
                    misses.append((q, None))
                    continue
                answered += 1
                if is_correct(picked[0], g):
                    correct += 1
                else:
                    misses.append((q, picked[0]))
            results.append(
                {
                    "variant": f"k={k:<2} title={'on ' if use_title else 'off'} {name}",
                    "correct": correct,
                    "answered": answered,
                    "wrong": answered - correct,
                    "misses": misses,
                }
            )

results.sort(key=lambda r: (-r["correct"], r["wrong"]))
print(f"\n{'variant':<26} {'correct':>7} {'answered':>9} {'wrong':>6}   (out of {len(questions)})")
for r in results:
    print(f"{r['variant']:<26} {r['correct']:>7} {r['answered']:>9} {r['wrong']:>6}")

print(f"\nMISSES FOR THE TOP VARIANT ({results[0]['variant'].strip()})")
for q, picked in results[0]["misses"]:
    print(f"  {q}  ->  {picked!r}")