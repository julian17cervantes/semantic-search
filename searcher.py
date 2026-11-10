import time

import rerank_settings as rs
from retrieval_lib import rerank_head


class Searcher:
    def __init__(self, model, index, chunks, reranker=None):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.reranker = reranker

    def _passage(self, i):
        row = self.chunks.iloc[i]
        if rs.RERANK_USE_TITLES:
            return f"{row['title']}. {row['text']}"
        return row["text"]

    def _retrieve(self, query, k, rerank):
        use_rerank = rerank and self.reranker is not None
        n = max(k, rs.RERANK_DEPTH) if use_rerank else k

        t0 = time.perf_counter()
        vector = self.model.encode([query], normalize_embeddings=True)
        t1 = time.perf_counter()
        scores, ids = self.index.search(vector, n)
        t2 = time.perf_counter()

        semantic = {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i >= 0}
        order = [int(i) for i in ids[0] if i >= 0]

        rerank_ms = None
        rerank_scores = {}
        if use_rerank:
            head = order[: rs.RERANK_DEPTH]
            t3 = time.perf_counter()
            raw = self.reranker.predict([(query, self._passage(i)) for i in head])
            rerank_ms = round((time.perf_counter() - t3) * 1000, 2)
            head_scores = [float(s) for s in raw]
            rerank_scores = dict(zip(head, head_scores))
            order = rerank_head(order, head_scores, rs.RERANK_DEPTH)

        hits = []
        for i in order[:k]:
            row = self.chunks.iloc[i]
            hit = {
                "title": row["title"],
                "url": row["url"],
                "score": round(semantic[i], 4),
                "text": row["text"],
            }
            if i in rerank_scores:
                hit["rerank_score"] = round(rerank_scores[i], 4)
            hits.append(hit)
        timings = {
            "encode_ms": round((t1 - t0) * 1000, 2),
            "search_ms": round((t2 - t1) * 1000, 2),
            "rerank_ms": rerank_ms,
        }
        return hits, timings

    def retrieve(self, query, k=5):
        """Plain semantic retrieval with full chunk text: (hits, encode_ms, search_ms)."""
        hits, timings = self._retrieve(query, k, rerank=False)
        return hits, timings["encode_ms"], timings["search_ms"]

    def search(self, query, k=5, rerank=False):
        hits, timings = self._retrieve(query, k, rerank)
        results = []
        for h in hits:
            item = {"title": h["title"], "url": h["url"], "score": h["score"], "snippet": h["text"][:300]}
            if "rerank_score" in h:
                item["rerank_score"] = h["rerank_score"]
            results.append(item)
        response = {
            "query": query,
            "results": results,
            "encode_ms": timings["encode_ms"],
            "search_ms": timings["search_ms"],
            "reranked": timings["rerank_ms"] is not None,
        }
        if response["reranked"]:
            response["rerank_ms"] = timings["rerank_ms"]
        return response