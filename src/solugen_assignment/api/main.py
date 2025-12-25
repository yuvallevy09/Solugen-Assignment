from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from solugen_assignment.api.models import (
    IndexBuildResponse,
    IndexProgressResponse,
    IndexEmbeddingSampleResponse,
    IndexStatusResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from solugen_assignment.api.retrieval import ChromaChunkIndex


try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can also come from the shell.
    pass

DEFAULT_PROCESSED_PATH = Path(
    os.getenv("PROCESSED_DATASET_JSONL", "data/processed/bbc_politics_news_under29900.jsonl")
)
DEFAULT_CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", "chroma_db"))
DEFAULT_COLLECTION = os.getenv("CHROMA_COLLECTION", "bbc_politics_chunks")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

_INDEX: ChromaChunkIndex | None = None

app = FastAPI(title="Solugen Assignment - Retrieval API", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def ui() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not found (missing static/index.html).")
    return FileResponse(index)


@app.get("/health")
def health() -> dict:
    return {"ok": True}

def _get_index() -> ChromaChunkIndex:
    path = DEFAULT_PROCESSED_PATH
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Processed dataset not found at '{path}'. "
                "Run scripts/prepare_bbc_dataset.py first or set PROCESSED_DATASET_JSONL."
            ),
        )
    global _INDEX
    if _INDEX is None:
        _INDEX = ChromaChunkIndex(
            dataset_path=path,
            persist_dir=DEFAULT_CHROMA_DIR,
            collection_name=DEFAULT_COLLECTION,
            embedding_model=DEFAULT_EMBED_MODEL,
        )
    return _INDEX


@app.get("/index/status", response_model=IndexStatusResponse)
def index_status() -> IndexStatusResponse:
    idx = _get_index()
    st = idx.status()
    return IndexStatusResponse(**st)


@app.post("/index/build", response_model=IndexBuildResponse)
def index_build() -> IndexBuildResponse:
    idx = _get_index()
    t0 = time.time()
    try:
        idx.ensure_built()
        st = idx.status()
        return IndexBuildResponse(ok=True, built=bool(st.get("built")), count=int(st.get("count", 0)))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        print(f"[index] build elapsed_s={time.time() - t0:.2f}", flush=True)


@app.get("/index/sample", response_model=IndexEmbeddingSampleResponse)
def index_sample(limit: int = 1) -> IndexEmbeddingSampleResponse:
    idx = _get_index()
    if not idx.is_built():
        raise HTTPException(
            status_code=409,
            detail="Index is not built yet. Call POST /index/build first.",
        )
    samples = idx.sample_embeddings(limit=limit)
    return IndexEmbeddingSampleResponse(samples=samples)


@app.get("/index/progress", response_model=IndexProgressResponse)
def index_progress() -> IndexProgressResponse:
    idx = _get_index()
    return IndexProgressResponse(**idx.progress())


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.time()
    print(
        f"[search] start top_k={req.top_k} threshold={req.similarity_threshold} q={req.query!r}",
        flush=True,
    )
    idx = _get_index()
    if not idx.is_built():
        raise HTTPException(
            status_code=409,
            detail="Index is not built yet. Call POST /index/build first.",
        )

    try:
        hits = idx.query(
            req.query,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        print(f"[search] done elapsed_s={time.time() - t0:.2f}", flush=True)

    results = [
        SearchResult(
            chunk_id=str(h.get("chunk_id", "")),
            doc_id=str(h.get("doc_id", "")),
            score=float(h.get("score", 0.0)),
            category=str(h.get("category", "")),
            kind=str(h.get("kind", "")),
            source_path=str(h.get("source_path", "")),
            start_char=(int(h["start_char"]) if h.get("start_char") is not None else None),
            end_char=(int(h["end_char"]) if h.get("end_char") is not None else None),
            char_len=len(str(h.get("text", ""))),
            text=str(h.get("text", "")),
        )
        for h in hits
    ]
    return SearchResponse(query=req.query, top_k=req.top_k, results=results)


