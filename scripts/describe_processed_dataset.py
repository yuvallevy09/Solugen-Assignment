#!/usr/bin/env python3
"""
Describe a processed JSONL dataset produced by scripts/prepare_bbc_dataset.py.

This prints a compact summary that is useful for the assignment writeup:
- number of docs, total chars, per-doc char/word stats
- doc_ids and titles

Stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional


def _pct(data: list[int], p: float) -> Optional[float]:
    if not data:
        return None
    xs = sorted(data)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    return xs[f] * (c - k) + xs[c] * (k - f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _title(text: str) -> str:
    return (text.split("\n", 1)[0] if text else "").strip()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Describe a processed dataset JSONL.")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/processed/bbc_politics_news_under29900.jsonl"),
        help="Path to processed JSONL (default: data/processed/bbc_politics_news_under29900.jsonl)",
    )
    args = parser.parse_args(argv)

    path: Path = args.jsonl
    if not path.exists():
        raise SystemExit(f"[ERROR] File not found: {path}")

    rows = _load_jsonl(path)
    if not rows:
        raise SystemExit(f"[ERROR] No rows in: {path}")

    char_lens = [int(r.get("char_len", 0)) for r in rows]
    word_counts = [len(str(r.get("text", "")).split()) for r in rows]

    # Sanity check: char_len matches actual length
    mismatches = [
        (str(r.get("doc_id", "")), int(r.get("char_len", 0)), len(str(r.get("text", ""))))
        for r in rows
        if int(r.get("char_len", 0)) != len(str(r.get("text", "")))
    ]

    print("POST-PROCESS DATASET SUMMARY")
    print(f"file: {path}")
    print(f"docs: {len(rows)}")
    print(f"total_chars(sum char_len): {sum(char_lens)}")
    print(
        "char_len min/median/mean/max:",
        min(char_lens),
        statistics.median(char_lens),
        round(statistics.mean(char_lens), 2),
        max(char_lens),
    )
    print("char_len p10/p90:", _pct(char_lens, 10), _pct(char_lens, 90))
    print(
        "word_count min/median/mean/max:",
        min(word_counts),
        statistics.median(word_counts),
        round(statistics.mean(word_counts), 2),
        max(word_counts),
    )
    print(f"mismatched char_len fields: {len(mismatches)}")
    if mismatches:
        print("first mismatch:", mismatches[0])

    print("\nPER-DOC (doc_id, char_len, words, title_line)")
    for r, words in zip(rows, word_counts):
        doc_id = str(r.get("doc_id", ""))
        title = _title(str(r.get("text", "")))
        print(f"{doc_id:>4}  chars={int(r.get('char_len', 0)):>5}  words={words:>4}  title={title!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


