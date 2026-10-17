import json
import statistics
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = "http://127.0.0.1:8000/search"
WARMUP = 20
REQUESTS = 200

titles = pd.read_parquet("data/chunks.parquet")["title"].drop_duplicates()
titles = titles.sample(WARMUP + REQUESTS, random_state=1).tolist()

encode, search, total = [], [], []
for n, title in enumerate(titles):
    url = BASE + "?" + urllib.parse.urlencode({"q": f"What is {title}?", "k": 5})
    start = time.perf_counter()
    with urllib.request.urlopen(url) as response:
        body = json.load(response)
    elapsed = (time.perf_counter() - start) * 1000
    if n >= WARMUP:
        encode.append(body["encode_ms"])
        search.append(body["search_ms"])
        total.append(elapsed)


def report(name, values):
    values = sorted(values)
    p50 = statistics.median(values)
    p95 = values[int(len(values) * 0.95) - 1]
    print(f"{name:>10}: p50 {p50:6.1f} ms   p95 {p95:6.1f} ms")


print(f"{REQUESTS} requests after {WARMUP} warm-up requests")
report("encode", encode)
report("search", search)
report("total", total)