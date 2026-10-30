import time


class Searcher:
    def __init__(self, model, index, chunks):
        self.model = model
        self.index = index
        self.chunks = chunks

    def retrieve(self, query, k=5):
        """Returns (hits with full chunk text, encode_ms, search_ms)."""
        t0 = time.perf_counter()
        vector = self.model.encode([query], normalize_embeddings=True)
        t1 = time.perf_counter()
        scores, ids = self.index.search(vector, k)
        t2 = time.perf_counter()

        hits = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            row = self.chunks.iloc[int(i)]
            hits.append(
                {
                    "title": row["title"],
                    "url": row["url"],
                    "score": round(float(score), 4),
                    "text": row["text"],
                }
            )
        return hits, round((t1 - t0) * 1000, 2), round((t2 - t1) * 1000, 2)

    def search(self, query, k=5):
        hits, encode_ms, search_ms = self.retrieve(query, k)
        results = [
            {
                "title": h["title"],
                "url": h["url"],
                "score": h["score"],
                "snippet": h["text"][:300],
            }
            for h in hits
        ]
        return {
            "query": query,
            "results": results,
            "encode_ms": encode_ms,
            "search_ms": search_ms,
        }