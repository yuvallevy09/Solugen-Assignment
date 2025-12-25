# Data layout

This project separates **raw** downloaded files from **processed** files that are ready for embedding.

## Folders

- `data/raw/`
  - Put the extracted Kaggle dataset here (do **not** commit it).
  - Example:
    - `data/raw/bbc-news-summary/` (whatever folder you get after unzip)
- `data/processed/`
  - Output of preparation scripts (small, curated, safe to commit).

## BBC dataset preparation

Use `scripts/prepare_bbc_dataset.py` to create a **politics-only** subset capped to **< 30,000 characters total**.


