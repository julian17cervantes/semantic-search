# Semantic Search

![Tests](https://github.com/julian17cervantes/semantic-search/actions/workflows/tests.yml/badge.svg)

A semantic search engine over Simple English Wikipedia, built end to end: data cleaning, chunking, embeddings, an approximate nearest-neighbor index, a FastAPI service, a benchmark against keyword search, and an extractive question-answering step that turns the retrieved passages into a short answer.

## Results

**Corpus:** 241,787 articles, cleaned down to 196,031 articles and split into 308,179 chunks of up to 1,000 characters.

### Earlier 27-query set

Measured on 27 hand-labeled questions. A hit means a chunk from the expected article appeared in the top results, with results deduplicated by article.

| Method | hit@1 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- |
| BM25 keyword search (with stemming) | 0.26 | 0.59 | 0.70 | 0.379 |
| Semantic search (all-MiniLM-L6-v2) | 0.59 | 0.85 | 0.96 | 0.702 |
| Hybrid (reciprocal rank fusion, untuned) | 0.41 | 0.81 | 0.89 | 0.584 |

Semantic search won clearly, and the service uses it. The equal-weight hybrid beat BM25 but not semantic search alone: compared with semantic search, it ranked the right article higher on 6 of the 27 queries, the same on 12, and lower on 9. I did not tune the fusion on these queries, to avoid fitting the test set.

## Retrieval evaluation v2

The 27-query set above was looked at many times while building, so I wrote two fresh sets before scoring anything: 60 development queries (`eval/retrieval_dev_all.csv`, written in two batches) and 30 test queries (`eval/retrieval_test.csv`). Each set mixes queries that use the article's own words, paraphrases, and definition-style questions. The only edits allowed after writing were fixing expected titles that do not exist in the corpus. The test set was scored once, after the settings were frozen. Intervals are 95% bootstrap intervals over queries.

**Test set (30 queries):**

| Method | hit@1 | hit@5 | hit@10 | MRR |
| --- | --- | --- | --- | --- |
| BM25 | 0.13 [0.03-0.27] | 0.57 [0.40-0.73] | 0.73 [0.57-0.90] | 0.29 [0.19-0.41] |
| Semantic | 0.63 [0.47-0.80] | 0.87 [0.73-0.97] | 0.93 [0.83-1.00] | 0.74 [0.61-0.87] |
| Hybrid (untuned fusion) | 0.43 [0.27-0.60] | 0.87 [0.73-0.97] | 0.90 [0.77-1.00] | 0.60 [0.46-0.74] |

### Experiment: article titles in the searched text

On 30 dev queries, adding the title to the embedded text raised semantic MRR by 0.059 (interval -0.009 to +0.145) but lowered hit@10 by one query. The adoption rule, fixed beforehand, required an MRR gain of at least 0.03 without lowering hit@10, so it was not met and the served index has no titles. The result is inconclusive.

### Experiment: cross-encoder reranking

A cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`, Apache-2.0) rescores the top 20 semantic results, reading the title and the chunk together with the query. Before running anything, I fixed these settings and an adoption rule: an MRR gain of at least 0.03, no drop in hit@10, and a median added latency under 100 ms.

| Semantic search + rerank | Semantic | + rerank | Change (95% interval) |
| --- | --- | --- | --- |
| **Dev, 60 queries:** MRR | 0.709 | 0.822 | +0.113 (+0.048 to +0.181) |
| **Test, 30 queries:** hit@1 | 0.633 | 0.767 | +0.133 (0.000 to +0.300) |
| Test: hit@5 | 0.867 | 0.933 | +0.067 (0.000 to +0.167) |
| Test: hit@10 | 0.933 | 0.967 | +0.033 (0.000 to +0.100) |
| Test: MRR | 0.741 | 0.830 | +0.089 (+0.009 to +0.180) |

On the test set, 8 queries ranked higher, 1 lower, and 21 were unchanged. The gain held on the unseen queries, but the lower end of the MRR interval is only +0.009, so it is confirmed narrowly, and the hit@k intervals include zero. A version without titles gave a smaller dev gain (MRR +0.070).

Reranking costs time: the reranking step took a median of 61 ms (p95 78 ms) per query on a laptop GPU, measured one query at a time, against about 6 ms for plain search. It is therefore opt-in: `GET /search?rerank=true`. The `/answer` endpoint does not use it, so the answer-step results above are unchanged.

**Limits of this evaluation**

- The sets are small (30 test queries), so intervals are wide. Most queries are well-known topics, and I wrote them myself.
- The 60 dev queries were used for several comparisons, so dev results are likely a little optimistic.
- A hit means the right article appeared, not that the passage answers the question.
- The reranker was trained on web-search data and is English-only. Latency was measured on one laptop, not under load.
- Reranking only reorders the top 20 results, so it cannot fix a query whose article is outside them (1 of the 30 test queries).

### Experiment: article titles in the searched text

Chunks often don't say what their article is about, because the title carries that. I tested adding the title to the text that is embedded and indexed for BM25, on 30 development queries (`eval/retrieval_dev.csv`). The rule for adopting it was fixed before the run: semantic MRR must rise by at least 0.03 without lowering hit@10.

| Semantic search | Without titles | With titles | Change (95% interval) |
| --- | --- | --- | --- |
| MRR | 0.640 | 0.699 | +0.059 (-0.009 to +0.145) |
| hit@10 | 0.967 | 0.933 | -0.033 (-0.100 to 0.000) |

MRR went up, but hit@10 dropped by one query, so the rule was not met and the service still uses the index without titles. With 30 queries, one query is 3.3 points, so this result is inconclusive. BM25 improved more clearly (MRR +0.062, interval +0.019 to +0.114), and the hybrid improved slightly, with intervals that include zero. The 30 test queries (`eval/retrieval_test.csv`) have not been scored.

### Index speed and accuracy

Measured on the full corpus against exact search, with 1,000 queries:

| Index | ms per query | recall@10 |
| --- | --- | --- |
| Exact (flat) | 4.51 | 1.000 |
| HNSW, efSearch=16 | 0.08 | 0.982 |
| HNSW, efSearch=32 | 0.13 | 0.992 |
| HNSW, efSearch=64 (used) | 0.22 | 0.997 |
| HNSW, efSearch=128 | 0.39 | 0.998 |

At efSearch=64, HNSW is about 20x faster than exact search while finding 99.7% of the exact top 10. Timings vary a little from run to run on a laptop (a re-run gave 4.48 and 0.22 ms; one run with other apps open was about 20% slower). Recall does not vary.

### Service latency

For `GET /search`, over 200 requests after 20 warm-up requests: median 6.1 ms and p95 10.5 ms end to end. Encoding the query took a median of 4.9 ms and the index search 0.4 ms.

## Answering questions

The search results can also be turned into a short answer with `GET /answer`. For a question, the system retrieves the top 10 chunks, runs a question-answering model on each one (`deepset/roberta-base-squad2`, licensed CC-BY-4.0) to pick an answer span, adds the article title to each chunk first, and adds up the scores of identical answers across chunks. An answer with a combined score under 0.5 is withheld ("no answer found"). Answers are labeled high confidence at 1.0 or above and low confidence in between.

The settings were chosen on 25 tuning questions and frozen in `answer_settings.py` before 30 new questions were written. The new questions were scored once, and nothing was changed afterward.

| Result on 30 held-out factoid questions | Count |
| --- | --- |
| Answered | 23 |
| Correct, among answered | 22 |
| High confidence | 16 answered, 16 correct |
| Low confidence | 7 answered, 6 correct |
| Withheld | 7 |

That is 22 of 30 correct overall, roughly 56% to 86% allowing for the small sample, and 22 of 23 correct among the answers given. Over 100 requests to GET /answer (one at a time, on a laptop GPU), the median was 193.3 ms and the p95 was 249.6 ms, of which reading the 10 chunks took a median of 186.7 ms.

## Limitations

- The retrieval test set is small (27 queries), so each query moves a score by about 4 points. I wrote the queries myself, and most are easy factual questions.
- A retrieval hit only means the right article was returned. It does not check that the returned passage actually answers the question.
- Neither the embeddings nor the BM25 index include article titles. Only the answer step adds them.
- Only the first 5 chunks of each article are indexed, and the embedding model reads about 256 tokens per chunk.
- `GET /search` returns chunks, so the same article can appear more than once in the results.
- Latency was measured on a laptop with one client on localhost, with the models on the Apple GPU (MPS). It is not a production figure.
- The answer test is 30 questions, so the result has a wide margin of uncertainty.
- The one wrong answer was `JPY` for the currency of Japan, where the gold answer was "yen". It is the ISO code for the yen, so it is arguably right, but the matching rule was fixed beforehand, so it counts as wrong.
- Seven test questions were withheld, including one that scored 0.49 against the 0.5 cutoff. Whether a lower cutoff would be better was not tested, because that would require tuning on the test set.
- In tuning, the three remaining wrong answers all came from "largest ..." questions. In the test, four of the eight superlative questions were answered correctly, four were withheld, and none were wrong.
- The reader picks short spans of text. It works for factoid questions and gives fragments for "why" and "how" questions.
- Answers can still be wrong, and a high-confidence label is not a guarantee.

## How it works

1. `download.py` fetches Simple English Wikipedia from Hugging Face.
2. `chunking.py` drops short articles, packs paragraphs into chunks, and drops tiny fragments.
3. `embed.py` embeds the chunks in resumable batches (about 11.5 minutes on a Mac laptop GPU).
4. `build_index.py` builds the exact index and the HNSW index, and `bench_index.py` measures recall and speed against the exact one.
5. `main.py` serves `GET /search` and `GET /answer`. The search response reports encode and search time, and the answer response adds the reading time.
6. `compare.py` evaluates BM25, semantic, and hybrid retrieval on `eval/queries.csv`.
7. `reader.py` finds an answer span in each chunk, and `answering.py` combines the spans into one answer and applies the confidence cutoff.
8. `tune_answers.py` compared reader settings on `eval/answers_dev.csv`, and `score_answers.py` scores a question file with the frozen settings. The one test run is saved in `results/answers_test.json`.

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
python build_index.py
uvicorn main:app
```

Then open http://127.0.0.1:8000/docs and try `GET /search?q=...` and `GET /answer?q=...`. The first run downloads the embedding model and the question-answering model (about 500 MB), and the server loads both at startup.

To reproduce the measurements:

```bash
python bench_index.py
python compare.py
python score_answers.py eval/answers_dev.csv
```

The data, embeddings, and indexes are large (the HNSW index is about 530 MB) and are not in the repo.

A note for Mac users: loading `faiss` before `torch` crashed with a segmentation fault on the machine this was built on, so the scripts that load the reader import `torch` first.

## API

| Method | Path | Description |
| --- | --- | --- |
| GET | /search?q=...&k=5 | Top matching chunks, with encode and search time |
| GET | /answer?q=... | A short answer, its source article, a confidence label, and timings |
| GET | /health | Health check |

## Tests

The tests use a fake model, index, and reader, so they don't need PyTorch or the real data:

```bash
pip install -r requirements-ci.txt
pytest
```

They cover the API layer, the chunking, the answer-matching rule, and the rule that combines answers. They do not cover the embedding step or the real models.

## License

The code is released under the MIT License (see `LICENSE`). The data and the question-answering model have their own licenses, described below.

## Data and credits

- **Data:** Simple English Wikipedia, from the `wikimedia/wikipedia` dataset on Hugging Face, licensed under CC BY-SA 3.0 and GFDL. Credit: Wikimedia Foundation.
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Question-answering model:** `deepset/roberta-base-squad2` by deepset, licensed CC-BY-4.0.