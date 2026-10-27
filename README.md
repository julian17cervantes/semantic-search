# Semantic Search

![Tests](https://github.com/julian17cervantes/semantic-search/actions/workflows/tests.yml/badge.svg)

A semantic search engine over Simple English Wikipedia, built end to end: data cleaning, chunking, embeddings, an approximate nearest-neighbor index, a FastAPI service, and a benchmark against keyword search.

## Results

**Corpus:** 241,787 articles, cleaned down to 196,031 articles and split into 308,179 chunks of up to 1,000 characters.

**Retrieval quality** on 27 hand-labeled questions. A hit means a chunk from the expected article appeared in the top results, with results deduplicated by article:

| Method | hit@1 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- |
| BM25 keyword search (with stemming) | 0.26 | 0.59 | 0.70 | 0.379 |
| Semantic search (all-MiniLM-L6-v2) | 0.59 | 0.85 | 0.96 | 0.702 |
| Hybrid (reciprocal rank fusion, untuned) | 0.41 | 0.81 | 0.89 | 0.584 |

Semantic search won clearly. The equal-weight hybrid beat BM25 but not semantic search alone, so the service uses semantic search. I did not tune the fusion on these queries, to avoid fitting the test set.

**Index speed and accuracy** on the full corpus, compared against exact search (1,000 queries):

| Index | ms per query | recall@10 |
| --- | --- | --- |
| Exact (flat) | 4.51 | 1.000 |
| HNSW, efSearch=16 | 0.08 | 0.982 |
| HNSW, efSearch=32 | 0.13 | 0.992 |
| HNSW, efSearch=64 (used) | 0.22 | 0.997 |
| HNSW, efSearch=128 | 0.39 | 0.998 |

**Service latency** for the FastAPI endpoint, over 200 requests after 20 warm-up requests: median 6.1 ms and p95 10.5 ms end to end. Encoding the query took a median of 4.9 ms and the index search 0.4 ms.

### Limitations

- The evaluation set is small (27 queries), so each query moves a score by about 4 points. I wrote the queries myself, and most are easy factual questions.
- A hit only means the right article was returned. It doesn't check that the returned passage actually answers the question.
- Latency was measured on a laptop with one client on localhost, with query encoding on the Apple GPU (MPS). It is not a production figure.
- Only the first 5 chunks of each article are indexed, and the embedding model reads about 256 tokens per chunk.

## How it works

1. `download.py` fetches Simple English Wikipedia from Hugging Face.
2. `chunking.py` drops short articles, packs paragraphs into chunks, and drops tiny fragments.
3. `embed.py` embeds the chunks in resumable batches (about 11.5 minutes on an Apple Silicon laptop).
4. `build_index.py` builds the exact index and HNSW indexes, and `bench_index.py` measures recall and speed against the exact one.
5. `main.py` serves `GET /search?q=...&k=5` and reports encode and search time for every request.
6. `compare.py` evaluates BM25, semantic, and hybrid search on `eval/queries.csv`.

## Run it yourself

```bash
git clone https://github.com/julian17cervantes/semantic-search.git
cd semantic-search
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python download.py
python chunking.py
python embed.py
python build_index.py # Builds the exact and HNSW indexes
python bench_index.py # optional: measures recall and speed
uvicorn main:app
```

Then open http://127.0.0.1:8000/docs. The data and indexes are large and are not in the repo.

## Tests

The tests use a fake model and index, so they don't need PyTorch or the real data:

```bash
pip install -r requirements-ci.txt
pytest
```

## Data

Simple English Wikipedia, from the `wikimedia/wikipedia` dataset on Hugging Face, licensed under CC BY-SA 3.0 and GFDL. Credit: Wikimedia Foundation.