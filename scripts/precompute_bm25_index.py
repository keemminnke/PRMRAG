#!/usr/bin/env python3
"""Pre-compute and cache a BM25 index for a JSONL corpus.

Intended for large corpora (e.g., KILT Wikipedia). The script:
- Loads documents from JSONL with fields {id,title,text}
- Normalizes text (joins list fields, strips newlines)
- Builds a BM25Okapi index
- Saves the pickled index to disk for reuse

Note: BM25 indexing keeps the tokenized corpus in memory. For multi-million
docs you need ample RAM; use --max-docs for smoke tests.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval import BM25Retriever  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and cache a BM25 index for a JSONL corpus."
    )
    parser.add_argument(
        "--corpus-file",
        type=Path,
        default=Path("data/raw/kilt_wikipedia_corpus.jsonl"),
        help="Path to corpus JSONL (expects id/title/text).",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=Path("data/embeddings/kilt_wikipedia_bm25.pkl"),
        help="Where to save the pickled BM25 index.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit number of docs (for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild index even if output already exists.",
    )
    return parser.parse_args()


def normalize_doc(raw: Dict[str, Any]) -> Dict[str, str]:
    """Convert raw JSON line to retriever-friendly schema."""
    text_field = raw.get("text", "")
    if isinstance(text_field, list):
        text = " ".join(part.replace("\n", " ").strip() for part in text_field if part)
    else:
        text = str(text_field).replace("\n", " ").strip()

    return {
        "id": str(raw.get("id") or raw.get("_id") or raw.get("wikipedia_id")),
        "title": str(raw.get("title") or raw.get("wikipedia_title") or "").strip(),
        "text": text,
    }


def load_corpus(path: Path, max_docs: Optional[int]) -> List[Dict[str, str]]:
    """Load up to max_docs documents from JSONL."""
    docs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus", unit="doc"):
            if max_docs is not None and len(docs) >= max_docs:
                break
            raw = json.loads(line)
            docs.append(normalize_doc(raw))
    return docs


def main():
    args = parse_args()

    if not args.corpus_file.exists():
        print(f"ERROR: Corpus file not found: {args.corpus_file}")
        sys.exit(1)

    if args.output_index.exists() and not args.overwrite:
        print(f"Index already exists at {args.output_index}. Use --overwrite to rebuild.")
        sys.exit(0)

    if args.output_index.exists() and args.overwrite:
        args.output_index.unlink()

    print(f"Corpus: {args.corpus_file}")
    print(f"Output index: {args.output_index}")
    if args.max_docs:
        print(f"Limiting to first {args.max_docs} docs (smoke test)")

    corpus = load_corpus(args.corpus_file, args.max_docs)
    print(f"Loaded {len(corpus):,} documents")

    args.output_index.parent.mkdir(parents=True, exist_ok=True)

    print("\nBuilding BM25 index (this may take a while)...")
    retriever = BM25Retriever(
        corpus=corpus,
        index_cache_path=str(args.output_index),
    )

    if args.output_index.exists():
        size_gb = args.output_index.stat().st_size / 1e9
        print(f"\n✅ Saved BM25 index to {args.output_index} (~{size_gb:.2f} GB)")
    else:
        print("\n⚠️ Failed to write BM25 index.")
        sys.exit(1)


if __name__ == "__main__":
    main()
