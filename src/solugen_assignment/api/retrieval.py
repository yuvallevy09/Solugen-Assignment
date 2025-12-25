from __future__ import annotations

import hashlib
import json
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import tiktoken
import httpx
from openai import DefaultHttpxClient
from openai import OpenAI


@dataclass(frozen=True)
class Doc:
    doc_id: str
    category: str
    kind: str
    char_len: int
    text: str
    source_path: str = ""


def load_processed_jsonl(path: Path) -> list[Doc]:
    docs: list[Doc] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Doc(
                    doc_id=str(obj.get("doc_id", "")),
                    category=str(obj.get("category", "")),
                    kind=str(obj.get("kind", "")),
                    char_len=int(obj.get("char_len", 0)),
                    text=str(obj.get("text", "")),
                    source_path=str(obj.get("source_path", "")),
                )
            )
    return docs


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    category: str
    kind: str
    source_path: str
    text: str
    start_char: int
    end_char: int


def _get_encoding(model: str) -> tiktoken.Encoding:
    # text-embedding-3-* uses cl100k_base; keep a safe fallback.
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(enc: tiktoken.Encoding, text: str) -> int:
    return len(enc.encode(text))


def _split_title_and_body(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text:
        return "", ""
    first_nl = text.find("\n")
    if first_nl == -1:
        return text.strip(), ""
    title = text[:first_nl].strip()
    body = text[first_nl + 1 :].strip()
    return title, body


def chunk_doc_paragraph_aware(
    doc: Doc,
    *,
    embedding_model: str,
    target_tokens: int = 400,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    """
    Paragraph-aware chunking:
    - split on blank lines (\n\n)
    - greedily merge paragraphs up to ~target_tokens
    - carry over tail paragraphs totalling ~overlap_tokens
    - prepend title to every chunk
    """
    enc = _get_encoding(embedding_model)

    title, body = _split_title_and_body(doc.text)

    # If the doc is effectively a single line/title (no body), still return one chunk.
    if title and not body:
        return [
            Chunk(
                chunk_id=f"{doc.doc_id}#c0",
                doc_id=doc.doc_id,
                category=doc.category,
                kind=doc.kind,
                source_path=doc.source_path,
                text=title.strip(),
                start_char=0,
                end_char=len(title),
            )
        ]
    body_paras = [p.strip() for p in body.split("\n\n") if p.strip()] if body else []
    if not body_paras:
        # Fall back to chunking the full text (still title-prefixed logic works)
        body_paras = [body] if body else []

    title_prefix = f"{title}\n\n" if title else ""

    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_start_para_idx = 0

    # For approximate char offsets, we walk through a reconstructed body with \n\n separators.
    # This is "best effort" for traceability; it is not used for retrieval.
    joined_body = "\n\n".join(body_paras)
    para_offsets: list[tuple[int, int]] = []
    offset = 0
    for i, p in enumerate(body_paras):
        start = offset
        end = start + len(p)
        para_offsets.append((start, end))
        offset = end + (2 if i < len(body_paras) - 1 else 0)

    def finalize_chunk(para_idx_end_inclusive: int) -> None:
        nonlocal cur, cur_start_para_idx
        if not cur:
            return
        chunk_text = title_prefix + "\n\n".join(cur)
        chunk_id = f"{doc.doc_id}#c{len(chunks)}"
        # Approx offsets within the body (exclude title).
        start_char = para_offsets[cur_start_para_idx][0] if para_offsets else 0
        end_char = para_offsets[para_idx_end_inclusive][1] if para_offsets else len(body)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                category=doc.category,
                kind=doc.kind,
                source_path=doc.source_path,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
            )
        )

    i = 0
    while i < len(body_paras):
        candidate = cur + [body_paras[i]]
        candidate_text = title_prefix + "\n\n".join(candidate)
        if _count_tokens(enc, candidate_text) <= target_tokens or not cur:
            if not cur:
                cur_start_para_idx = i
            cur = candidate
            i += 1
            continue

        # Finalize current chunk at i-1.
        finalize_chunk(i - 1)

        # Build overlap: keep last paragraphs totalling ~overlap_tokens (within body only).
        tail: list[str] = []
        tail_tokens = 0
        j = i - 1
        while j >= cur_start_para_idx and tail_tokens < overlap_tokens:
            para = body_paras[j]
            # Count tokens without title to approximate overlap tokens.
            t = _count_tokens(enc, para)
            tail.insert(0, para)
            tail_tokens += t
            j -= 1

        cur = tail
        cur_start_para_idx = max(j + 1, 0) if tail else i

    # Final chunk
    if cur:
        finalize_chunk(len(body_paras) - 1)

    # Ensure we never return empty chunks
    return [c for c in chunks if c.text.strip()]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _index_fingerprint(
    *,
    dataset_path: Path,
    collection_base: str,
    collection_effective: str,
    embedding_model: str,
    target_tokens: int,
    overlap_tokens: int,
) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "collection": {"base": collection_base, "effective": collection_effective},
        "embedding_model": embedding_model,
        "chunking": {
            "strategy": "paragraph_aware_title_prefixed",
            "target_tokens": target_tokens,
            "overlap_tokens": overlap_tokens,
        },
        "generated_at": time.time(),
    }


class ChromaChunkIndex:
    def __init__(
        self,
        *,
        dataset_path: Path,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str,
        target_tokens: int = 400,
        overlap_tokens: int = 80,
    ) -> None:
        self.dataset_path = dataset_path
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.persist_dir / "index_meta.json"

        self._chroma = chromadb.PersistentClient(path=str(self.persist_dir))
        # If an index was previously built, reuse its "effective" collection name.
        effective = self.collection_name
        if self._meta_path.exists():
            try:
                existing = json.loads(self._meta_path.read_text(encoding="utf-8"))
                effective = (
                    (existing.get("collection") or {}).get("effective")  # type: ignore[union-attr]
                    or self.collection_name
                )
            except Exception:
                effective = self.collection_name
        self._collection = self._chroma.get_or_create_collection(
            name=str(effective),
            metadata={"hnsw:space": "cosine"},
        )

        self._progress_lock = threading.Lock()
        self._progress: dict[str, Any] = {
            "stage": "idle",  # idle|reusing|chunking|embedding|persisting|done|error
            "started_at": None,
            "updated_at": None,
            "done": False,
            "error": None,
            "docs_total": 0,
            "docs_done": 0,
            "chunks_total": 0,
            "chunks_embedded": 0,
            "batches_total": 0,
            "batch_num": 0,
            "message": "",
        }

        # NOTE: We intentionally initialize the OpenAI client lazily.
        # This keeps status/debug endpoints usable even before an API key is configured.
        self._openai: OpenAI | None = None

    def _get_openai(self) -> OpenAI:
        if self._openai is not None:
            return self._openai
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        # Fail fast if the network / key is misconfigured instead of hanging for minutes.
        # Use an explicit httpx client so timeouts are reliably enforced.
        http_client = DefaultHttpxClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        self._openai = OpenAI(api_key=api_key, http_client=http_client, max_retries=2)
        return self._openai

    def _progress_update(self, **fields: Any) -> None:
        now = time.time()
        with self._progress_lock:
            if self._progress.get("started_at") is None:
                self._progress["started_at"] = now
            self._progress["updated_at"] = now
            self._progress.update(fields)

    def progress(self) -> dict[str, Any]:
        with self._progress_lock:
            return dict(self._progress)

    def _meta_matches(self, meta: dict[str, Any]) -> bool:
        if not self._meta_path.exists():
            return False
        try:
            existing = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        # Compare stable keys only
        keys = ["dataset_path", "dataset_sha256", "embedding_model", "chunking", "collection"]
        return all(existing.get(k) == meta.get(k) for k in keys)

    @staticmethod
    def _effective_collection_name(*, base: str, desired_meta: dict[str, Any]) -> str:
        """
        Avoid delete/recreate hangs by writing each fingerprint to its own collection.
        This keeps builds deterministic and makes the build step robust.
        """
        stable = {
            "dataset_path": desired_meta.get("dataset_path"),
            "dataset_sha256": desired_meta.get("dataset_sha256"),
            "embedding_model": desired_meta.get("embedding_model"),
            "chunking": desired_meta.get("chunking"),
        }
        h = hashlib.sha256(json.dumps(stable, sort_keys=True).encode("utf-8")).hexdigest()[:10]
        return f"{base}__{h}"

    def _desired_meta(self) -> dict[str, Any]:
        if not self.dataset_path.exists():
            raise RuntimeError(f"Processed dataset not found at '{self.dataset_path}'.")
        # First compute dataset sha etc; then derive the effective collection name deterministically.
        base_meta = {
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": _sha256_file(self.dataset_path),
            "embedding_model": self.embedding_model,
            "chunking": {
                "strategy": "paragraph_aware_title_prefixed",
                "target_tokens": self.target_tokens,
                "overlap_tokens": self.overlap_tokens,
            },
        }
        effective = self._effective_collection_name(base=self.collection_name, desired_meta=base_meta)
        return _index_fingerprint(
            dataset_path=self.dataset_path,
            collection_base=self.collection_name,
            collection_effective=effective,
            embedding_model=self.embedding_model,
            target_tokens=self.target_tokens,
            overlap_tokens=self.overlap_tokens,
        )

    def is_built(self) -> bool:
        """
        Return True if an index exists on disk for the current dataset + parameters and the
        collection is non-empty.
        """
        try:
            desired = self._desired_meta()
        except Exception:
            return False
        if not self._meta_matches(desired):
            return False
        try:
            return int(self._collection.count()) > 0
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        """
        Introspection helper for debugging + demos (does not call OpenAI).
        """
        existing_meta: dict[str, Any] | None = None
        if self._meta_path.exists():
            try:
                existing_meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except Exception:
                existing_meta = None
        try:
            desired_meta = self._desired_meta()
        except Exception as e:
            desired_meta = {"error": str(e)}
        try:
            count = int(self._collection.count())
        except Exception:
            count = 0
        return {
            "built": self.is_built(),
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "count": count,
            "existing_meta": existing_meta,
            "desired_meta": desired_meta,
        }

    def sample_embeddings(self, *, limit: int = 1) -> list[dict[str, Any]]:
        """
        Return up to `limit` sample rows including stored embedding vectors, for verifying that
        embeddings were actually written into Chroma.
        """
        n = max(1, min(int(limit), 3))
        out = self._collection.get(
            limit=n,
            include=["embeddings", "metadatas", "documents"],
        )
        ids = out.get("ids", []) or []
        docs = out.get("documents", []) or []
        metas = out.get("metadatas", []) or []
        embs = out.get("embeddings", []) or []

        samples: list[dict[str, Any]] = []
        for i in range(min(len(ids), len(embs))):
            emb = embs[i] or []
            dim = len(emb)
            # Basic numeric sanity checks (avoid heavy numpy dependency).
            norm_sq = 0.0
            head = []
            for j, v in enumerate(emb):
                fv = float(v)
                if j < 8:
                    head.append(fv)
                norm_sq += fv * fv
            samples.append(
                {
                    "chunk_id": ids[i],
                    "doc_id": (metas[i] or {}).get("doc_id", ""),
                    "embedding_dim": dim,
                    "embedding_norm": norm_sq ** 0.5,
                    "embedding_head": head,
                    "text_preview": (docs[i] or "")[:220],
                    "metadata": metas[i] or {},
                }
            )
        return samples

    def ensure_built(self) -> None:
        """
        Ensure the persistent Chroma collection is built for the current dataset + parameters.
        If anything changes (dataset hash, chunk params, model), we rebuild.
        """
        self._progress_update(
            stage="idle",
            done=False,
            error=None,
            message="starting",
            docs_total=0,
            docs_done=0,
            chunks_total=0,
            chunks_embedded=0,
            batches_total=0,
            batch_num=0,
        )

        desired_meta = self._desired_meta()
        if self._meta_matches(desired_meta) and int(self._collection.count()) > 0:
            print("[index] reuse existing index (fingerprint match)", flush=True)
            self._progress_update(stage="reusing", done=True, message="reused existing index")
            return

        # Rebuild into a deterministic "effective" collection name; avoid delete_collection hangs.
        effective = (desired_meta.get("collection") or {}).get("effective") or self.collection_name
        print(f"[index] building index into collection={effective!r}", flush=True)
        self._progress_update(stage="chunking", message=f"building (chunking) -> {effective}")
        self._collection = self._chroma.get_or_create_collection(
            name=str(effective),
            metadata={"hnsw:space": "cosine"},
        )

        docs = load_processed_jsonl(self.dataset_path)
        self._progress_update(docs_total=len(docs), docs_done=0)
        chunks: list[Chunk] = []
        for n, d in enumerate(docs, start=1):
            chunks.extend(
                chunk_doc_paragraph_aware(
                    d,
                    embedding_model=self.embedding_model,
                    target_tokens=self.target_tokens,
                    overlap_tokens=self.overlap_tokens,
                )
            )
            self._progress_update(docs_done=n, message=f"chunking {n}/{len(docs)} docs")

        if not chunks:
            raise RuntimeError("No chunks produced from processed dataset.")
        print(f"[index] chunks: {len(chunks)}", flush=True)
        self._progress_update(chunks_total=len(chunks), message=f"chunked {len(chunks)} chunks")

        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "category": c.category,
                "kind": c.kind,
                "source_path": c.source_path,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        # Embed in batches to keep memory bounded.
        self._progress_update(stage="embedding", message="embedding chunks")
        embeddings: list[list[float]] = []
        batch_size = 64
        total_batches = (len(texts) + batch_size - 1) // batch_size
        self._progress_update(batches_total=total_batches, batch_num=0, chunks_embedded=0)
        openai = self._get_openai()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            print(
                f"[index] embedding batch {batch_num}/{total_batches} "
                f"(items {i + 1}-{min(i + len(batch), len(texts))} of {len(texts)})",
                flush=True,
            )
            self._progress_update(
                batch_num=batch_num,
                chunks_embedded=i,
                message=f"embedding batch {batch_num}/{total_batches}",
            )
            try:
                resp = openai.embeddings.create(model=self.embedding_model, input=batch)
            except Exception as e:
                self._progress_update(stage="error", error=str(e), done=True, message="embedding failed")
                raise RuntimeError(
                    "OpenAI embeddings call failed while building the index. "
                    "Check OPENAI_API_KEY, network access, and whether your account/project allows embeddings."
                ) from e
            embeddings.extend([d.embedding for d in resp.data])
            self._progress_update(chunks_embedded=min(i + len(batch), len(texts)))

        self._progress_update(stage="persisting", message="writing to Chroma")
        self._collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        self._meta_path.write_text(json.dumps(desired_meta, ensure_ascii=False, indent=2) + "\n")
        self._progress_update(stage="done", done=True, message="index built")

    def query(
        self,
        query: str,
        *,
        top_k: int,
        similarity_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        q = query.strip()
        if not q:
            return []

        if not self.is_built():
            raise RuntimeError(
                "Index is not built for the current dataset/parameters. "
                "Build it first (POST /index/build) or set it up ahead of time."
            )

        try:
            print("[search] embedding query", flush=True)
            resp = self._get_openai().embeddings.create(model=self.embedding_model, input=[q])
        except Exception as e:
            raise RuntimeError(
                "OpenAI embeddings call failed for the query. Check OPENAI_API_KEY and network access."
            ) from e
        q_emb = resp.data[0].embedding

        out = self._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "ids"],
        )

        ids = out.get("ids", [[]])[0]
        docs = out.get("documents", [[]])[0]
        metas = out.get("metadatas", [[]])[0]
        dists = out.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []
        for chunk_id, text, meta, dist in zip(ids, docs, metas, dists):
            # With cosine space, Chroma returns distance ~= (1 - cosine_similarity).
            score = 1.0 - float(dist)
            if similarity_threshold is not None and score < similarity_threshold:
                continue
            results.append(
                {
                    "chunk_id": str(chunk_id),
                    "score": score,
                    "text": str(text),
                    **(meta or {}),
                }
            )
        return results


