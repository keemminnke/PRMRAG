#!/usr/bin/env python3
"""Prepare KILT knowledge source with BGE-M3 embeddings.

This script:
- Streams the KILT knowledge source dump (JSONL) without loading it fully.
- Normalizes each document to {id, title, text}.
- Optionally writes a slimmed corpus JSONL for downstream use.
- Writes BGE-M3 embeddings to a float16 .npy (np.memmap) file to save RAM.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from FlagEmbedding import BGEM3FlagModel
except ImportError:
    print("ERROR: Missing dependency 'FlagEmbedding'. Install with `pip install FlagEmbedding`.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream KILT knowledge source and generate BGE-M3 embeddings."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/kilt/kilt_knowledgesource.json",
        help="Path to the KILT knowledge source JSONL.",
    )
    parser.add_argument(
        "--output-corpus",
        type=str,
        default="data/raw/kilt_wikipedia_corpus.jsonl",
        help="Where to write the normalized corpus JSONL (id, title, text).",
    )
    parser.add_argument(
        "--output-embeddings",
        type=str,
        default="data/embeddings/kilt_wikipedia_bge_m3.npy",
        help="Where to write the BGE-M3 embeddings (.npy, float16).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of documents to encode per batch (adjust for GPU RAM).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for the encoder (BGE-M3 default is 512).",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Skip counting: pre-set the number of docs for memmap allocation.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit docs to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-corpus-save",
        action="store_true",
        help="Do not write the normalized corpus JSONL (saves disk space).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if they are present.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-m3",
        help="BGE model name.",
    )
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Store embeddings as float32 instead of float16 (larger but lossless).",
    )
    return parser.parse_args()


def normalize_doc(raw: Dict[str, Any]) -> Dict[str, str]:
    """Convert raw KILT doc to the retriever-friendly schema."""
    text_field = raw.get("text", "")
    if isinstance(text_field, list):
        text = " ".join(part.replace("\n", " ").strip() for part in text_field if part)
    else:
        text = str(text_field).replace("\n", " ").strip()

    doc_id = raw.get("_id") or raw.get("id") or raw.get("wikipedia_id") or ""

    return {
        "id": str(doc_id),
        "title": str(raw.get("wikipedia_title") or raw.get("title") or "").strip(),
        "text": text,
    }


def count_documents(path: Path) -> int:
    """Count documents in a large JSONL file."""
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in tqdm(f, desc="Counting documents", unit="doc"):
            count += 1
    return count


def encode_batch(
    model: BGEM3FlagModel,
    docs: List[Dict[str, str]],
    max_length: int,
) -> np.ndarray:
    """Encode a batch of documents."""
    texts = [f"{doc['title']} {doc['text']}".strip() for doc in docs]
    return model.encode(
        texts,
        max_length=max_length,
        batch_size=len(texts),
    )["dense_vecs"]


def write_batch(
    memmap: Optional[np.memmap],
    embed_dim: Optional[int],
    emb_path: Path,
    target_docs: int,
    embeddings: np.ndarray,
    start_idx: int,
    dtype,
) -> Tuple[np.memmap, int]:
    """Initialize memmap if needed and write a batch slice."""
    if embed_dim is None:
        embed_dim = embeddings.shape[1]
        memmap = np.memmap(
            emb_path,
            dtype=dtype,
            mode="w+",
            shape=(target_docs, embed_dim),
        )

    end_idx = start_idx + embeddings.shape[0]
    memmap[start_idx:end_idx] = embeddings.astype(dtype)
    return memmap, embed_dim


def main():
    args = parse_args()

    input_path = Path(args.input)
    corpus_out = Path(args.output_corpus)
    emb_out = Path(args.output_embeddings)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    if emb_out.exists() and not args.overwrite:
        print(f"Embeddings already exist at {emb_out}. Use --overwrite to rebuild.")
        sys.exit(0)

    if corpus_out.exists() and not args.skip_corpus_save and not args.overwrite:
        print(f"Corpus file already exists at {corpus_out}. Use --overwrite to rebuild.")
        sys.exit(0)

    # Determine how many docs to process.
    target_docs = args.max_docs
    if target_docs is None:
        target_docs = args.num_docs if args.num_docs is not None else count_documents(input_path)

    print(f"\nPreparing to encode {target_docs:,} documents")
    print(f"  - Input: {input_path}")
    if not args.skip_corpus_save:
        print(f"  - Normalized corpus: {corpus_out}")
    print(f"  - Embeddings: {emb_out}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Model: {args.model_name}")

    # Set up model and outputs.
    dtype = np.float32 if args.force_fp32 else np.float16
    model = BGEM3FlagModel(args.model_name, use_fp16=not args.force_fp32)

    emb_out.parent.mkdir(parents=True, exist_ok=True)
    corpus_f = None
    if not args.skip_corpus_save:
        corpus_out.parent.mkdir(parents=True, exist_ok=True)
        corpus_f = open(corpus_out, "w", encoding="utf-8")

    memmap: Optional[np.memmap] = None
    embed_dim: Optional[int] = None
    written = 0
    docs_seen = 0
    buffer: List[Dict[str, str]] = []

    progress_total = target_docs

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=progress_total, desc="Encoding KILT", unit="doc"):
                if docs_seen >= target_docs:
                    break

                raw_doc = json.loads(line)
                doc = normalize_doc(raw_doc)
                docs_seen += 1

                if corpus_f:
                    corpus_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

                buffer.append(doc)

                if len(buffer) >= args.batch_size:
                    embeddings = encode_batch(model, buffer, args.max_length)
                    memmap, embed_dim = write_batch(
                        memmap,
                        embed_dim,
                        emb_out,
                        target_docs,
                        embeddings,
                        written,
                        dtype,
                    )
                    written += embeddings.shape[0]
                    buffer.clear()

            # Flush any remaining docs.
            if buffer and written < target_docs:
                if written + len(buffer) > target_docs:
                    buffer = buffer[: target_docs - written]
                embeddings = encode_batch(model, buffer, args.max_length)
                memmap, embed_dim = write_batch(
                    memmap,
                    embed_dim,
                    emb_out,
                    target_docs,
                    embeddings,
                    written,
                    dtype,
                )
                written += embeddings.shape[0]
                buffer.clear()
    finally:
        if corpus_f:
            corpus_f.close()
        if memmap is not None:
            memmap.flush()

    if memmap is None or embed_dim is None:
        print("No documents were processed. Nothing to write.")
        sys.exit(1)

    print("\n✅ Done")
    print(f"  - Documents encoded: {written:,}")
    print(f"  - Embeddings shape: ({written}, {embed_dim}) dtype={dtype}")
    print(f"  - Embeddings file: {emb_out} (~{emb_out.stat().st_size / 1e9:.2f} GB)")
    if not args.skip_corpus_save:
        print(f"  - Normalized corpus: {corpus_out} (~{corpus_out.stat().st_size / 1e9:.2f} GB)")
    print("\nUse embedding_cache_path pointing to the .npy when initializing BGERetriever.")


if __name__ == "__main__":
    main()
