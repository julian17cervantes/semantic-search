import os
import json

from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time

import torch
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

import answer_settings as cfg
from answer_metrics import is_correct
from answering import combine, passages_for_reading
from reader import SpanReader

path = sys.argv[1]
chunks = pd.read_parquet("data/chunks.parquet")
rows = pd.read_csv(path)
questions = rows["question"].tolist()
golds = [a.split("|") for a in rows["answer"]]

index = faiss.read_index("indexes/hnsw.faiss")
faiss.ParameterSpace().set_index_parameter(index, "efSearch", 64)
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "mps" if torch.backends.mps.is_available() else "cpu"
reader = SpanReader(device=device)
reader.read("warm up", ["warm up text"])

outcomes, start = [], time.perf_counter()
for question, gold in zip(questions, golds):
    vector = encoder.encode([question], normalize_embeddings=True)
    _, ids = index.search(vector, cfg.READ_TOP_K)
    sub = chunks.iloc[ids[0]]
    titles = sub["title"].tolist()
    reads = reader.read(question, passages_for_reading(titles, sub["text"].tolist()))
    result = combine(reads, titles)
    correct = result["answer"] is not None and is_correct(result["answer"], gold)
    outcomes.append({"question": question, **result, "correct": correct})
ms = (time.perf_counter() - start) * 1000 / len(questions)

total = len(outcomes)
answered = [o for o in outcomes if o["answer"] is not None]
correct = sum(o["correct"] for o in answered)
print(f"{path}: {total} questions, {ms:.0f} ms per question including encoding and search")
print(f"answered {len(answered)}, correct {correct}, wrong {len(answered) - correct}")
for level in ("high", "low", "none"):
    group = [o for o in outcomes if o["confidence"] == level]
    print(f"  confidence {level:>4}: {len(group):>2} questions, {sum(o['correct'] for o in group):>2} correct")

print("\nWRONG OR WITHHELD")
for o in outcomes:
    if not o["correct"]:
        print(f"  [{o['confidence']}] {o['score']:.2f}  {o['answer']!r}  {o['question']}")

if len(sys.argv) > 2:
    summary = {
        "file": path,
        "questions": total,
        "answered": len(answered),
        "correct": correct,
        "wrong": len(answered) - correct,
        "ms_per_question": round(ms),
        "settings": {
            "read_top_k": cfg.READ_TOP_K,
            "use_titles": cfg.USE_TITLES,
            "no_answer_below": cfg.NO_ANSWER_BELOW,
            "confident_from": cfg.CONFIDENT_FROM,
        },
        "outcomes": outcomes,
    }
    Path(sys.argv[2]).parent.mkdir(exist_ok=True)
    Path(sys.argv[2]).write_text(json.dumps(summary, indent=2, default=str))
    print("saved", sys.argv[2])