#!/usr/bin/env python3
"""Download and prepare HotpotQA Full Wiki corpus with BGE-M3 embeddings.

This script:
1. Downloads HotpotQA dataset from Hugging Face
2. Extracts unique Wikipedia paragraphs from fullwiki setting
3. Converts to JSONL format compatible with BGERetriever
4. Generates BGE-M3 embeddings (takes ~2-4 hours on GPU)
5. Saves embeddings cache for fast retrieval

Note: Uses Hugging Face datasets library for reliable download.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from collections import OrderedDict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval import BGERetriever

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not found.")
    print("Install it with: pip install datasets")
    sys.exit(1)


def extract_corpus_from_hotpotqa(output_file: Path):
    """Extract unique Wikipedia paragraphs from HotpotQA fullwiki dataset.

    Downloads HotpotQA from Hugging Face and extracts all unique paragraphs
    from the context field to build a corpus.

    Returns:
        Number of unique documents extracted
    """
    print("Downloading HotpotQA dataset from Hugging Face...")
    print("This may take a few minutes...")

    # Load the validation fullwiki split
    dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
    print(f"✓ Loaded {len(dataset)} examples")

    # Extract unique paragraphs
    print("\nExtracting unique paragraphs from context...")
    unique_paragraphs = OrderedDict()  # title -> set of paragraphs
    doc_id = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Processing"):
            # Each example has 'context' field with {'title': [...], 'sentences': [[...]]}
            titles = example['context']['title']
            sentences_list = example['context']['sentences']

            for title, sentences in zip(titles, sentences_list):
                # Join sentences to form paragraph
                paragraph = ' '.join(sentences).strip()

                if not paragraph:
                    continue

                # Create unique key
                key = f"{title}:::{paragraph[:100]}"  # Use first 100 chars as fingerprint

                if key not in unique_paragraphs:
                    unique_paragraphs[key] = True

                    doc = {
                        'id': f'wiki_{doc_id}',
                        'title': title,
                        'text': paragraph
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                    doc_id += 1

    print(f"✓ Extracted {doc_id:,} unique documents to {output_file}")
    return doc_id


def build_embeddings(corpus_file: Path, output_embeddings: Path, batch_size: int = 64):
    """Build BGE-M3 embeddings for corpus."""
    print(f"\n{'='*70}")
    print("BUILDING BGE-M3 EMBEDDINGS")
    print(f"{'='*70}\n")

    # Load corpus
    print(f"Loading corpus from {corpus_file}...")
    corpus = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading"):
            corpus.append(json.loads(line))

    print(f"✓ Loaded {len(corpus):,} documents")

    # Initialize retriever (this will build embeddings)
    print(f"\nInitializing BGE-M3 retriever...")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  - Embedding cache: {output_embeddings}")
    print(f"\nThis will take ~2-4 hours depending on GPU...")

    output_embeddings.parent.mkdir(parents=True, exist_ok=True)

    retriever = BGERetriever(
        corpus=corpus,
        batch_size=batch_size,
        embedding_cache_path=str(output_embeddings)
    )

    print(f"\n✓ Embeddings saved to {output_embeddings}")
    print(f"  - Shape: {retriever.embeddings.shape}")
    print(f"  - Size: {output_embeddings.stat().st_size / 1e9:.2f} GB")

    return retriever


def main():
    # Paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    embeddings_dir = data_dir / "embeddings"

    raw_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Files
    corpus_jsonl = raw_dir / "hotpotqa_fullwiki_corpus.jsonl"
    embeddings_file = embeddings_dir / "hotpotqa_fullwiki_bge_m3.npy"

    print("="*70)
    print("HOTPOTQA FULL WIKI CORPUS SETUP (via Hugging Face)")
    print("="*70)
    print(f"\nThis script will:")
    print(f"1. Download HotpotQA fullwiki dataset from Hugging Face")
    print(f"2. Extract unique Wikipedia paragraphs (~100K-200K docs)")
    print(f"3. Convert to JSONL format")
    print(f"4. Generate BGE-M3 embeddings (takes 1-2 hours on GPU)")
    print(f"\nData will be saved to:")
    print(f"  - Corpus: {corpus_jsonl}")
    print(f"  - Embeddings: {embeddings_file}")
    print("="*70)

    # Check if already exists
    if corpus_jsonl.exists() and embeddings_file.exists():
        print(f"\n⚠️  Files already exist!")
        print(f"  - {corpus_jsonl}")
        print(f"  - {embeddings_file}")
        response = input("\nDo you want to rebuild? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    input("\nPress Enter to start, or Ctrl+C to cancel...")

    # Step 1 & 2: Download and extract corpus
    if not corpus_jsonl.exists():
        print(f"\n[1/2] Downloading and extracting corpus from HotpotQA...")
        num_docs = extract_corpus_from_hotpotqa(corpus_jsonl)
    else:
        print(f"\n[1/2] Using existing corpus: {corpus_jsonl}")
        with open(corpus_jsonl, 'r') as f:
            num_docs = sum(1 for _ in f)
        print(f"  - Documents: {num_docs:,}")

    # Step 3: Build embeddings
    if not embeddings_file.exists():
        print(f"\n[2/2] Building BGE-M3 embeddings...")
        build_embeddings(corpus_jsonl, embeddings_file, batch_size=64)
    else:
        print(f"\n[2/2] Using existing embeddings: {embeddings_file}")

    # Summary
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE!")
    print(f"{'='*70}")
    print(f"\nCorpus: {corpus_jsonl}")
    print(f"  - Documents: {num_docs:,}")
    print(f"  - Size: {corpus_jsonl.stat().st_size / 1e6:.2f} MB")
    print(f"\nEmbeddings: {embeddings_file}")
    if embeddings_file.exists():
        print(f"  - Size: {embeddings_file.stat().st_size / 1e9:.2f} GB")
    print(f"\nYou can now run tests with Full Wiki corpus:")
    print(f"  python scripts/batch_test_adaptive.py --num-questions 10")
    print()


if __name__ == "__main__":
    main()
