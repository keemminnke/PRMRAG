#!/usr/bin/env python3
"""Pre-compute and save BGE-M3 embeddings for corpus documents.

This script encodes the entire corpus once and saves the embeddings to disk,
significantly speeding up subsequent retriever initialization.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval import BGERetriever


def load_corpus(corpus_file: str):
    """Load corpus documents from JSONL file."""
    corpus = []
    with open(corpus_file, 'r') as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute BGE-M3 embeddings for corpus"
    )
    parser.add_argument(
        "--corpus-file",
        type=str,
        required=True,
        help="Path to corpus JSONL file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save embeddings (.npy file)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-m3",
        help="BGE model name (default: BAAI/bge-m3)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("PRE-COMPUTE BGE-M3 EMBEDDINGS")
    print("=" * 80)
    print(f"\nCorpus file: {args.corpus_file}")
    print(f"Output file: {args.output_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model: {args.model_name}")

    # Load corpus
    print(f"\n[1] Loading corpus from {args.corpus_file}...")
    corpus = load_corpus(args.corpus_file)
    print(f"✓ Loaded {len(corpus)} documents")

    # Initialize retriever (this will encode and save embeddings)
    print(f"\n[2] Encoding corpus with BGE-M3...")
    print(f"    This will take a while for large corpora...")
    retriever = BGERetriever(
        corpus=corpus,
        model_name=args.model_name,
        batch_size=args.batch_size,
        embedding_cache_path=args.output_file,
    )

    print(f"\n{'=' * 80}")
    print("DONE!")
    print(f"{'=' * 80}")
    print(f"\nEmbeddings saved to: {args.output_file}")
    print(f"Shape: {retriever.doc_embeddings.shape}")
    print(f"\nTo use these embeddings, pass embedding_cache_path='{args.output_file}'")
    print(f"when initializing BGERetriever.")
    print()


if __name__ == "__main__":
    main()
