# RAG evaluation plan (written before any scoring)

**Candidates:** `gemma3:4b` and `llama3.1:8b` through Ollama (temperature 0, seed 0, at most 120 new tokens), picked from a four-case smoke test.

**Frozen:** the prompt (`rag.SYSTEM_PROMPT`) and the context: 5 sources per question, taken from the top 20 semantic results reranked by the frozen cross-encoder with titles (`rerank_settings.py`). The prompt does not change after this file is committed.

## Question sets

| Set | File | Size |
| --- | --- | --- |
| Dev, answerable | `eval/answers_dev.csv` (also used for the extractive reader) | 25 |
| Dev, unanswerable | `eval/rag_dev_unanswerable.csv` | 15 |
| Test, answerable | `eval/rag_test_answerable.csv` (fresh) | 30 |
| Test, unanswerable | `eval/rag_test_unanswerable.csv` (fresh) | 15 |

## Metrics

- **Correct:** the reply is not an abstention, and a gold answer appears in it as whole words (case, punctuation, and articles ignored).
- **Grounded correct:** correct, and at least one valid cited source contains the gold answer.
- **Abstention on unanswerable questions:** the reply begins with the fallback sentence. Anything else counts as an unsupported answer.
- **Citation validity:** the share of non-abstained answers with at least one valid citation and no invalid ones.
- **Latency:** median and p95 end to end per question, after one warm-up call per model.

## Model selection (dev only)

1. Highest grounded-correct count on the 25 answerable dev questions.
2. Ties: more abstentions on the 15 unanswerable dev questions.
3. Ties: the smaller model.

## Test (once)

Score the selected model on both test sets, and score the frozen extractive reader (`score_answers.py`) on the same 30 answerable questions. Report everything with 95% intervals, whichever way it falls.

## Unanswerable questions that get answered anyway

Each non-abstaining reply to an unanswerable question is read by hand. If the cited source really contains a correct answer, that question is dropped from the unanswerable count for every system, and the dropped questions are listed in the README.

## Allowed edits to the question files after this commit

Only fixing a gold answer that the corpus check shows is absent. Nothing after seeing model outputs.