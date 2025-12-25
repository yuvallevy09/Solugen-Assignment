from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query to retrieve relevant context for.")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks/documents to return.")
    similarity_threshold: float | None = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum similarity score (cosine similarity) to keep results.",
    )


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    category: str
    kind: str
    source_path: str = ""
    start_char: int | None = None
    end_char: int | None = None
    char_len: int
    text: str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: list[SearchResult]


class IndexBuildResponse(BaseModel):
    ok: bool
    built: bool
    count: int


class IndexStatusResponse(BaseModel):
    built: bool
    collection_name: str
    persist_dir: str
    count: int
    existing_meta: dict | None = None
    desired_meta: dict


class IndexEmbeddingSample(BaseModel):
    chunk_id: str
    doc_id: str
    embedding_dim: int
    embedding_norm: float
    embedding_head: list[float]
    text_preview: str
    metadata: dict


class IndexEmbeddingSampleResponse(BaseModel):
    samples: list[IndexEmbeddingSample]


class IndexProgressResponse(BaseModel):
    stage: str
    started_at: float | None = None
    updated_at: float | None = None
    done: bool
    error: str | None = None
    docs_total: int
    docs_done: int
    chunks_total: int
    chunks_embedded: int
    batches_total: int
    batch_num: int
    message: str

