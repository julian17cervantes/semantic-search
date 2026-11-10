import json
import statistics
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = "http://127.0.0.1:8000/search"
WARMUP = 10
REQUESTS = 100

queries = pd.read_csv("eval/retrieval_dev_all.csv")["query"].tolist()
cycle = [queries[i % len(queries)] for i in range(WARMUP + REQUESTS)]


def call(query, rerank):
    url = BASE + "?" + urllib.parse.urlencode({"q": query, "k": 5, "rerank": str(rerank).lower()})
    start = time.perf_counter()
    with urllib.request.urlopen(url) as response:
        body = json.load(response)
    elapsed = (time.perf_counter() - start) * 1000
    assert body["reranked"] == rerank, f"expected reranked={rerank}, got {body['reranked']}"
    return body, elapsed


results = {False: {"encode": [], "search": [], "rerank": [], "total": []},
           True: {"encode": [], "search": [], "rerank": [], "total": []}}
for n, query in enumerate(cycle):
    order = (False, True) if n % 2 == 0 else (True, False)
    for rerank in order:
        body, elapsed = call(query, rerank)
        if n >= WARMUP:
            r = results[rerank]
            r["encode"].append(body["encode_ms"])
            r["search"].append(body["search_ms"])
            r["rerank"].append(body.get("rerank_ms", 0.0))
            r["total"].append(elapsed)


def report(name, values):
    values = sorted(values)
    p50 = statistics.median(values)
    p95 = values[int(len(values) * 0.95) - 1]
    print(f"{name:>8}: p50 {p50:7.1f} ms   p95 {p95:7.1f} ms")


print(f"{REQUESTS} requests per mode after {WARMUP} warm-up requests, one at a time")
for rerank, label in ((False, "plain search"), (True, "rerank=true")):
    print(f"\n{label}")
    for part in ("encode", "search", "rerank", "total"):
        if part == "rerank" and not rerank:
            continue
        report(part, results[rerank][part])