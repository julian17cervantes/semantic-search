from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from searcher import Searcher

from fastapi.responses import FileResponse

from answerer import Answerer

import rerank_settings as rs

EF_SEARCH = 64


@asynccontextmanager
async def lifespan(app):
    import torch
    import faiss
    import pandas as pd
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from reader import SpanReader

    chunks = pd.read_parquet("data/chunks.parquet")
    index = faiss.read_index("indexes/hnsw.faiss")
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.encode(["warm up"])

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    reranker = CrossEncoder(rs.RERANK_MODEL, device=device)
    reranker.predict([("warm up", "warm up text")])
    searcher = Searcher(model, index, chunks, reranker=reranker)

    reader = SpanReader(device=device)
    reader.read("warm up", ["warm up text"])

    app.state.searcher = searcher
    app.state.answerer = Answerer(searcher, reader)
    yield


app = FastAPI(title="Semantic Search", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(
    request: Request,
    q: str = Query(..., max_length=500),
    k: int = Query(5, ge=1, le=20),
    rerank: bool = Query(
        rs.RERANK_DEFAULT_ON,
        description="Rescore the top chunks with a cross-encoder (about 60 ms extra on the benchmark laptop)",
    ),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    searcher = request.app.state.searcher
    if rerank and searcher.reranker is None:
        raise HTTPException(status_code=400, detail="Reranking is not available on this server")
    return searcher.search(q.strip(), k, rerank=rerank)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")

@app.get("/answer")
def answer(request: Request, q: str = Query(..., max_length=500)):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    return request.app.state.answerer.answer(q.strip())