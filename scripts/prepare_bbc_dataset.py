#!/usr/bin/env python3
"""
Prepare the Kaggle "BBC News Summary" dataset to match the home-exam constraints.

Outputs a small, curated subset (default: politics) capped at < 30,000 characters total.

This script is intentionally stdlib-only so it works without extra dependencies.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CATEGORY_ALIASES = {
    "business": "business",
    "entertainment": "entertainment",
    "politics": "politics",
    "sport": "sport",
    "sports": "sport",
    "tech": "tech",
    "technology": "tech",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_category(s: str) -> str:
    s = s.strip().lower()
    if s in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[s]
    raise ValueError(
        f"Unknown category '{s}'. Use one of: {', '.join(sorted(set(CATEGORY_ALIASES.values())))}"
    )


def _read_text_file(path: Path) -> str:
    # Kaggle text files are usually utf-8, but we keep it robust.
    return path.read_text(encoding="utf-8", errors="ignore")


_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving paragraph breaks.
    This helps stabilize character counts and reduces accidental noise.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [(_MULTISPACE.sub(" ", ln)).strip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    text = _MULTINEWLINE.sub("\n\n", text)
    return text


def _infer_category_from_path(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    for part in parts:
        if part in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[part]
    return None


def _infer_kind_from_path(path: Path) -> str:
    """
    Best-effort inference of whether a file is a full news article or a summary.
    The Kaggle dataset often includes two trees: 'News Articles' and 'Summaries'.
    """
    parts = [p.lower() for p in path.parts]
    if any(p in {"summaries", "summary", "news_summary", "news summaries"} for p in parts):
        return "summary"
    if any(p in {"news articles", "news", "articles", "news_articles"} for p in parts):
        return "news"
    return "unknown"


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _sort_key_doc_id(doc_id: str) -> Tuple[int, str]:
    n = _safe_int(doc_id)
    if n is None:
        return (1_000_000_000, doc_id)
    return (n, doc_id)


@dataclass(frozen=True)
class DocRef:
    doc_id: str
    category: str
    kind: str  # "news" | "summary"
    path: Path


def _scan_txt_files(raw_root: Path) -> List[DocRef]:
    txt_files = [p for p in raw_root.rglob("*.txt") if p.is_file()]
    refs: List[DocRef] = []
    for p in txt_files:
        category = _infer_category_from_path(p)
        if category is None:
            continue
        kind = _infer_kind_from_path(p)
        if kind == "unknown":
            # Still keep it; we might only have one tree extracted.
            kind = "news"
        refs.append(DocRef(doc_id=p.stem, category=category, kind=kind, path=p))

    # Kaggle zips are sometimes extracted into duplicate nested folder trees
    # (e.g., both ".../News Articles/..." and ".../BBC News Summary/News Articles/...").
    # Deduplicate by (category, kind, doc_id), keeping the shortest (most direct) path.
    best: Dict[Tuple[str, str, str], DocRef] = {}
    for r in refs:
        key = (r.category, r.kind, r.doc_id)
        prev = best.get(key)
        if prev is None:
            best[key] = r
            continue
        # Prefer the path with fewer segments; if tie, prefer lexicographically smallest.
        if len(r.path.parts) < len(prev.path.parts) or (
            len(r.path.parts) == len(prev.path.parts) and str(r.path) < str(prev.path)
        ):
            best[key] = r

    return list(best.values())


def _pick_kind_available(refs: List[DocRef], prefer: str) -> str:
    kinds = sorted(set(r.kind for r in refs))
    if prefer != "auto":
        if prefer in kinds:
            return prefer
        raise ValueError(f"Requested --source {prefer} not found. Found kinds: {kinds}")
    # auto
    if "news" in kinds:
        return "news"
    if "summary" in kinds:
        return "summary"
    return kinds[0]


def _select_docs(
    refs: List[DocRef],
    selection: str,
    seed: int,
    max_total_chars: int,
    max_docs: Optional[int],
) -> Tuple[List[Dict], Dict]:
    """
    Greedy selection: iterate candidates (in deterministic or randomized order),
    include docs that fit under the character cap.
    """
    candidates = list(refs)
    if selection == "first":
        candidates.sort(key=lambda r: _sort_key_doc_id(r.doc_id))
    elif selection == "random":
        rnd = random.Random(seed)
        rnd.shuffle(candidates)
    else:
        raise ValueError("selection must be one of: first, random")

    chosen: List[Dict] = []
    total_chars = 0
    skipped_too_large = 0
    skipped_empty = 0

    for r in candidates:
        if max_docs is not None and len(chosen) >= max_docs:
            break

        raw_text = _read_text_file(r.path)
        text = _clean_text(raw_text)
        if not text:
            skipped_empty += 1
            continue

        char_len = len(text)
        if total_chars + char_len > max_total_chars:
            skipped_too_large += 1
            continue

        chosen.append(
            {
                "doc_id": r.doc_id,
                "category": r.category,
                "kind": r.kind,
                "source_path": str(r.path),
                "char_len": char_len,
                "text": text,
            }
        )
        total_chars += char_len

    meta = {
        "generated_at": _now_iso(),
        "selection": selection,
        "seed": seed,
        "max_total_chars": max_total_chars,
        "max_docs": max_docs,
        "chosen_docs": len(chosen),
        "total_chars": total_chars,
        "skipped_empty": skipped_empty,
        "skipped_over_cap": skipped_too_large,
    }
    return chosen, meta


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare BBC dataset subset under a character cap.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw"),
        help="Folder containing the extracted Kaggle dataset (default: data/raw).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Where to write processed outputs (default: data/processed).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="politics",
        help="Category to keep (default: politics).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "news", "summary"],
        help="Prefer full news articles or summaries (default: auto, prefers news if present).",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="first",
        choices=["first", "random"],
        help="How to pick files under the cap (default: first).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (used only if --selection random).",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=29_900,
        help="Total character budget across all selected documents (default: 29900).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of documents (default: no cap).",
    )

    args = parser.parse_args(argv)
    raw_root: Path = args.raw_root
    out_dir: Path = args.out_dir

    if not raw_root.exists():
        print(
            f"[ERROR] raw root '{raw_root}' does not exist.\n"
            "Put the extracted Kaggle dataset under data/raw/ (example: data/raw/bbc-news-summary/...)",
            file=sys.stderr,
        )
        return 2

    category = _normalize_category(args.category)
    all_refs = _scan_txt_files(raw_root)
    if not all_refs:
        print(
            f"[ERROR] No .txt files found under '{raw_root}'.\n"
            "Make sure you downloaded + extracted the Kaggle dataset into data/raw/.\n"
            "Expected a folder tree with category names like 'politics', 'business', 'sport', 'tech'.",
            file=sys.stderr,
        )
        return 2

    refs_cat = [r for r in all_refs if r.category == category]
    if not refs_cat:
        found_categories = sorted(set(r.category for r in all_refs))
        print(
            f"[ERROR] No files found for category '{category}'. Found categories: {found_categories}",
            file=sys.stderr,
        )
        return 2

    kind = _pick_kind_available(refs_cat, args.source)
    refs_kind = [r for r in refs_cat if r.kind == kind]

    chosen, meta = _select_docs(
        refs=refs_kind,
        selection=args.selection,
        seed=args.seed,
        max_total_chars=args.max_total_chars,
        max_docs=args.max_docs,
    )

    if not chosen:
        print(
            f"[ERROR] Selected 0 docs for category='{category}', kind='{kind}' under "
            f"{args.max_total_chars} chars.\n"
            "Try increasing --max-total-chars, using --source summary, or changing --selection.",
            file=sys.stderr,
        )
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"bbc_{category}_{kind}_under{args.max_total_chars}.jsonl"
    out_meta = out_dir / f"bbc_{category}_{kind}_under{args.max_total_chars}.meta.json"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_full = {
        "dataset": "kaggle/pariza/bbc-news-summary",
        "raw_root": str(raw_root),
        "out_jsonl": str(out_jsonl),
        "category": category,
        "kind": kind,
        "files_scanned_total": len(all_refs),
        "files_in_category_total": len(refs_cat),
        "files_in_category_kind_total": len(refs_kind),
        **meta,
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta_full, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("[OK] Prepared dataset subset")
    print(f" - out:  {out_jsonl}")
    print(f" - meta: {out_meta}")
    print(f" - docs: {meta_full['chosen_docs']}")
    print(f" - total_chars: {meta_full['total_chars']} (cap: {args.max_total_chars})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


