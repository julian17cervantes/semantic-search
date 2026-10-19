from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from searcher import Searcher

from fastapi.responses import FileResponse

EF_SEARCH = 64


@asynccontextmanager
async def lifespan(app):
    import faiss
    import pandas as pd
    from sentence_transformers import SentenceTransformer

    chunks = pd.read_parquet("data/chunks.parquet")
    index = faiss.read_index("indexes/hnsw.faiss")
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", EF_SEARCH)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model.encode(["warm up"])
    app.state.searcher = Searcher(model, index, chunks)
    yield


app = FastAPI(title="Semantic Search", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/search")
def search(request: Request, q: str = Query(..., max_length=500), k: int = Query(5, ge=1, le=20),):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return request.app.state.searcher.search(q.strip(), k)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")