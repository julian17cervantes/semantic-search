import time


class Searcher:
    def __init__(self, model, index, chunks):
        self.model = model
        self.index = index
        self.chunks = chunks

    def search(self, query, k=5):
        t0 = time.perf_counter()
        vector = self.model.encode([query], normalize_embeddings=True)
        t1 = time.perf_counter()
        scores, ids = self.index.search(vector, k)
        t2 = time.perf_counter()

        results = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0:
                continue
            row = self.chunks.iloc[int(i)]
            results.append({
                    "title": row["title"],
                    "url": row["url"],
                    "score": round(float(score), 4),
                    "snippet": row["text"][:300],
                }
            )
        return {"query": query,
            "results": results,
            "encode_ms": round((t1 - t0) * 1000, 2),
            "search_ms": round((t2 - t1) * 1000, 2),
        }