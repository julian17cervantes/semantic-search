import json
import statistics
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = "http://127.0.0.1:8000/answer"
WARMUP = 5
REQUESTS = 100

questions = pd.read_csv("eval/answers_dev.csv")["question"].tolist()
cycle = [questions[i % len(questions)] for i in range(WARMUP + REQUESTS)]

parts = {"encode_ms": [], "search_ms": [], "read_ms": [], "total_ms": []}
for n, question in enumerate(cycle):
    url = BASE + "?" + urllib.parse.urlencode({"q": question})
    start = time.perf_counter()
    with urllib.request.urlopen(url) as response:
        body = json.load(response)
    elapsed = (time.perf_counter() - start) * 1000
    if n >= WARMUP:  # ignore the first requests while things warm up
        for key in ("encode_ms", "search_ms", "read_ms"):
            parts[key].append(body[key])
        parts["total_ms"].append(elapsed)


def report(name, values):
    values = sorted(values)
    p50 = statistics.median(values)
    p95 = values[int(len(values) * 0.95) - 1]
    print(f"{name:>10}: p50 {p50:7.1f} ms   p95 {p95:7.1f} ms")


print(f"{REQUESTS} requests after {WARMUP} warm-up requests, one at a time")
for key in ("encode_ms", "search_ms", "read_ms", "total_ms"):
    report(key.replace("_ms", ""), parts[key])