# Solugen Assignment — Retrieval (RAG) Component

This repo is set up to help you build the **retrieval pipeline** required by the assignment.

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
uv run uvicorn solugen_assignment.api.main:app --reload --port 8000
```

### 4) Environment variables

Copy `example.env` to `.env` and fill in your key:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `CHROMA_PERSIST_DIR` (default: `chroma_db`)
- `CHROMA_COLLECTION` (default: `bbc_politics_chunks`)

Test:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/docs
```

Search (retrieval; requires index built):

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"maternity pay rise", "top_k": 5}'
```

Search (embedding + Chroma chunk retrieval):

> Requires `OPENAI_API_KEY` and the embedding/vector-db deps installed.
> Build the index once (embeddings + Chroma persistence), then search.

```bash
curl -s -X POST http://127.0.0.1:8000/index/build

curl -s -X POST http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"maternity pay rise", "top_k": 5, "similarity_threshold": 0.2}'
```

Notes:

- Results are **chunks**, not full documents.
- Each result includes a `chunk_id` like `003#c1`, plus the chunk `text` and a cosine-similarity `score`.
- To verify embeddings exist in the vector DB, call:

```bash
curl -s http://127.0.0.1:8000/index/status
curl -s http://127.0.0.1:8000/index/sample
```

## Chunking & retrieval parameters (what we chose + why)

### Chunking strategy

Implemented in `src/solugen_assignment/api/retrieval.py` (`chunk_doc_paragraph_aware`):

- **Strategy**: paragraph-aware chunking (split on blank lines), greedily merging paragraphs up to a token target
- **Chunk size**: ~**400 tokens** (`target_tokens=400`)
- **Overlap**: ~**80 tokens** (`overlap_tokens=80`) by carrying over the tail paragraphs into the next chunk
- **Title handling**: the article title is **prepended to every chunk** to improve retrieval for title/entity-focused queries

Rationale:

- **Paragraph boundaries preserve coherence** better than arbitrary fixed-size splits for news articles.
- **400 tokens** is usually enough to capture a complete thought/claim + supporting sentences, while staying small for embeddings.
- **80-token overlap** reduces “boundary loss” when key facts span paragraphs.

### Retrieval parameters

- **Top‑K**: `top_k` (default **5**, allowed range **1–20**)
- **Similarity threshold**: optional `similarity_threshold` (cosine similarity). When set, results below the threshold are filtered out.
- **Score definition**: stored vectors use cosine space; Chroma returns a distance roughly \(1 - \text{cosine\_similarity}\), so the API reports `score = 1 - distance`.

## Minimal UI

After starting the API, open:

- `http://127.0.0.1:8000/`

This page only performs retrieval and displays the returned context chunks (no LLM / no generation).

## Dataset: BBC News Summary (Kaggle)

We use the Kaggle dataset **`pariza/bbc-news-summary`** and curate it to meet the exam constraint:

- **Total text under 30,000 characters**
- Single category (default: **politics**)

### Why this dataset

- **Realistic retrieval**: News articles contain enough context to make chunking, overlap, and top‑k meaningful (facts often span multiple sentences/paragraphs).
- **Good query diversity**: The same entities/topics recur across articles, so semantic retrieval has to distinguish “similar but not the same” passages.
- **Fits constraints + low cost**: We curate a small subset under 30k characters, keeping embedding cost far below the $1 limit.

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

- `data/raw/bbc-news-summary/`

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


