# Solugen Assignment — Retrieval (RAG) Component

This repo implements the **retrieval component** of a RAG system: take a user query, retrieve the most relevant text chunks from a small dataset, and show them in a minimal UI.

What this repo does:
- Builds an index by chunking documents, creating **OpenAI embeddings**, and storing them in **ChromaDB**
- Retrieves **top‑K** chunks for a query, with an optional **similarity threshold**
- Shows retrieved chunks + similarity scores in a simple browser UI

What this repo intentionally does **not** do:
- No LLM / chatbot / answer generation (the UI only displays retrieved context)

## Python setup (uv) + FastAPI

This project uses **[uv](https://github.com/astral-sh/uv)** for Python dependency management.

### 1) Install uv

Follow the official install instructions, then verify:

```bash
uv --version
```

### 2) Create venv + install deps

From the repo root:

```bash
uv sync
```

### 3) Run the API

```bash
uv run uvicorn solugen_assignment.api.main:app --reload --port 8001
```

### 4) Environment variables

Copy `example.env` to `.env` and fill in your key:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `CHROMA_PERSIST_DIR` (default: `chroma_db`)
- `CHROMA_COLLECTION` (default: `bbc_politics_chunks`)
- `PROCESSED_DATASET_JSONL` (default: `data/processed/bbc_politics_news_under29900.jsonl`)

Test:

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/docs
```

### 5) Build the index (embeddings) once

You can do this from the UI (button) or via API:

```bash
curl -s -X POST http://127.0.0.1:8001/index/build
```

### 6) Search (retrieval)

Search requires the index to be built first:

```bash
curl -s -X POST http://127.0.0.1:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"maternity pay rise", "top_k": 5}'
```

With a similarity threshold:

```bash
curl -s -X POST http://127.0.0.1:8001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"maternity pay rise", "top_k": 5, "similarity_threshold": 0.2}'
```

Notes:

- Results are **chunks**, not full documents.
- Each result includes a `chunk_id` like `003#c1`, plus the chunk `text` and a cosine-similarity `score`.
- To verify embeddings exist in the vector DB, call:

```bash
curl -s http://127.0.0.1:8001/index/status
curl -s http://127.0.0.1:8001/index/sample
```

## Minimal UI

After starting the API, open:

- `http://127.0.0.1:8001/`

Flow:
- Click **Build / Rebuild Index** once (stores embeddings in Chroma)
- Run searches and review the retrieved chunks + scores

## Chunking & retrieval parameters (what we chose + why)

### Chunking strategy

Implemented in `src/solugen_assignment/api/retrieval.py` (`chunk_doc_paragraph_aware`):

- **Strategy**: paragraph-aware chunking (split on blank lines), greedily merging paragraphs up to a token target
- **Chunk size**: ~**400 tokens** (`target_tokens=400`)
- **Overlap**: ~**80 tokens** (`overlap_tokens=80`) by carrying over the tail paragraphs into the next chunk
- **Title handling**: the article title is **prepended to every chunk** to improve retrieval for title/entity-focused queries

Rationale:

- **Paragraph boundaries preserve coherence** better than arbitrary fixed-size splits for news articles.
- **400 tokens** is usually enough to capture a complete thought + supporting sentences, while staying small enough to embed cheaply.
- **80-token overlap** reduces “boundary loss” when key facts span paragraphs.

### Retrieval parameters

- **Top‑K**: `top_k` (default **5**, allowed range **1–20**)
- **Similarity threshold**: optional `similarity_threshold` (cosine similarity). When set, results below the threshold are filtered out.
- **Score definition**: stored vectors use cosine space; Chroma returns a distance roughly \(1 - \text{cosine\_similarity}\), so the API reports `score = 1 - distance`.

## Vector DB choice: why Chroma

I picked **Chroma** because it’s the quickest way to ship a small, local retrieval system end‑to‑end:
- It runs locally (no external service setup), but still behaves like a “real” vector DB (collections, metadata, persistence).
- It supports cosine similarity out of the box, and persists to disk so you can build once and reuse.
- For a dataset this small, it’s more than enough and keeps the project simple and reproducible.

## Dataset: BBC News Summary (Kaggle)

We use the Kaggle dataset **`pariza/bbc-news-summary`** and curate it to meet the exam constraint:

- **Total text under 30,000 characters**
- Single category (default: **politics**)

### Why this dataset

I wanted something that feels like a real retrieval problem, but still fits the “very small dataset” constraint:
- News articles have enough structure (titles + paragraphs) to make chunking/overlap matter.
- Entities repeat across articles (people, parties, topics), so semantic search has to separate “related” from “actually relevant”.
- By curating to <30k characters total, embedding cost stays comfortably under the $1 limit.

### Expected user questions

- **Topic-focused**: “What did Labour propose about maternity pay?”
- **Entity-focused**: “What did Patricia Hewitt say about maternity leave?”
- **Event/decision**: “What happened in Gordon Brown’s Budget discussion?”
- **Comparison**: “Which articles mention maternity pay vs flexible working?”
- **Detail lookup**: “How much was maternity pay proposed to rise by?”

### 1) Download + extract the dataset

1. Download from Kaggle (manually) and unzip it.
2. Put the extracted folder under:

- `data/raw/`

Example:

- `data/raw/BBC News Summary/`

> `data/raw/` is gitignored on purpose.

### 2) Generate a compliant subset (<30k chars)

Run:

```bash
python3 scripts/prepare_bbc_dataset.py --raw-root data/raw --out-dir data/processed --category politics
```

Outputs:

- `data/processed/bbc_politics_news_under29900.jsonl` (or summary, depending on what’s available)
- `data/processed/bbc_politics_news_under29900.meta.json` (character counts + selection info)

Optional: print a quick “post-filter” dataset summary:

```bash
python3 scripts/describe_processed_dataset.py --jsonl data/processed/bbc_politics_news_under29900.jsonl
```

### Processed dataset details (what the API currently uses)

The committed processed dataset in this repo is:

- `data/processed/bbc_politics_news_under29900.jsonl`

It contains:

- **Docs**: 13 (all `politics`, all `news`)
- **Total characters**: 29,815 (cap: 29,900)
- **Doc IDs**: `001`–`012` plus `249`
- **Per-doc char length**: min/median/mean/max = 811 / 2518 / 2293.46 / 3197
- **Per-doc word count** (rough, whitespace split): min/median/mean/max = 142 / 428 / 390.92 / 536

Selection metadata from `data/processed/bbc_politics_news_under29900.meta.json`:

- **Files scanned**: 4,450 `.txt`
- **Politics files (all kinds)**: 834
- **Politics + news**: 417
- **Chosen docs**: 13
- **Skipped because adding would exceed cap**: 404

Included documents (doc_id → title):

- `001` → Labour plans maternity pay rise
- `002` → Watchdog probes e-mail deletions
- `003` → Hewitt decries 'career sexism'
- `004` → Labour chooses Manchester
- `005` → Brown ally rejects Budget spree
- `006` → 'Errors' doomed first Dome sale
- `007` → Fox attacks Blair's Tory 'lies'
- `008` → Women MPs reveal sexist taunts
- `009` → Campbell: E-mail row 'silly fuss'
- `010` → Crucial decision on super-casinos
- `011` → Mrs Howard gets key election role
- `012` → PM apology over jailings
- `249` → No election TV debate, says Blair

### Notes

- The script auto-detects dataset layouts as long as the extracted paths contain category folder names like `politics`, `business`, `sport`, `tech`.
- If you want reproducible random sampling under the cap:

```bash
python3 scripts/prepare_bbc_dataset.py --raw-root data/raw --out-dir data/processed --category politics --selection random --seed 42
```

## Cost note

This project embeds a very small amount of text (a curated subset under 30k characters, then chunked). With `text-embedding-3-small`, the total embedding cost is **well under $1** for typical runs.

## Troubleshooting

- **`POST /search` returns 409**: you need to run `POST /index/build` once first (or click the UI build button).
- **`OPENAI_API_KEY is not set`**: copy `example.env` to `.env` and set a real key (or export it in your shell).
- **No matches**: try lowering/clearing `similarity_threshold`, or increase `top_k`.


