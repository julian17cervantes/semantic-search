import threading
import time

import answer_settings as cfg
from answering import combine, passages_for_reading


class Answerer:
    def __init__(self, searcher, reader):
        self.searcher = searcher
        self.reader = reader
        self._lock = threading.Lock()

    def answer(self, question):
        hits, encode_ms, search_ms = self.searcher.retrieve(question, cfg.READ_TOP_K)
        titles = [h["title"] for h in hits]

        start = time.perf_counter()
        if hits:
            passages = passages_for_reading(titles, [h["text"] for h in hits])
            with self._lock:
                reads = self.reader.read(question, passages)
        else:
            reads = []
        read_ms = (time.perf_counter() - start) * 1000

        result = combine(reads, titles)
        source = None
        if result["source"] is not None:
            url = next(h["url"] for h in hits if h["title"] == result["source"])
            source = {"title": result["source"], "url": url}

        return {
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "score": result["score"],
            "source": source,
            "retrieved": [
                {"title": h["title"], "url": h["url"], "score": h["score"]} for h in hits[:5]
            ],
            "encode_ms": encode_ms,
            "search_ms": search_ms,
            "read_ms": round(read_ms, 2),
        }