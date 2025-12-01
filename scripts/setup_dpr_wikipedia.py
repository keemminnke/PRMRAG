#!/usr/bin/env python3
"""Download and prepare Meta DPR Wikipedia corpus with BGE-M3 embeddings.

This script:
1. Downloads DPR Wikipedia corpus (21M passages) from HuggingFace
2. Converts to JSONL format compatible with BGERetriever
3. Generates BGE-M3 embeddings (takes ~4-6 hours on GPU)
4. Saves embeddings cache for fast retrieval

Note: Uses facebook/wiki_dpr from HuggingFace (standard for Open-Domain QA).
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval import BGERetriever

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not found.")
    print("Install it with: pip install datasets")
    sys.exit(1)


def download_and_convert_corpus(corpus_file: Path):
    """Download DPR Wikipedia corpus and convert to JSONL format.

    Args:
        corpus_file: Output JSONL file path

    Returns:
        Number of documents saved
    """
    print("\n" + "="*70)
    print("DOWNLOADING DPR WIKIPEDIA CORPUS")
    print("="*70)
    print("\nDataset: facebook/wiki_dpr")
    print("Configuration: psgs_w100.no_embeddings")
    print("Size: 21M passages from Wikipedia (Dec 2018)")
    print("\nThis may take 30-60 minutes depending on internet speed...")
    print("="*70)

    # Load dataset WITHOUT pre-computed DPR embeddings (we'll use BGE-M3)
    # Note: We only need the text, not the DPR embeddings
    print("\n[1/2] Downloading corpus from HuggingFace...")

    # Use no_index.no_embeddings version (text only, no faiss, no embeddings)
    dataset = load_dataset(
        "facebook/wiki_dpr",
        "psgs_w100.nq.no_index.no_embeddings",
        split="train",
        trust_remote_code=True
    )

    print(f"✓ Loaded {len(dataset):,} passages")

    # Convert to JSONL format
    print(f"\n[2/2] Converting to JSONL format...")
    corpus_file.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_file, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(tqdm(dataset, desc="Converting")):
            # DPR format: id, text, title (and optionally embeddings)
            doc_data = {
                'id': str(doc['id']) if 'id' in doc else f'dpr_{i}',
                'title': doc['title'],
                'text': doc['text']
            }
            f.write(json.dumps(doc_data, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(dataset):,} documents to {corpus_file}")
    print(f"  - File size: {corpus_file.stat().st_size / 1e9:.2f} GB")

    return len(dataset)


def build_embeddings(corpus_file: Path, output_embeddings: Path, batch_size: int = 64):
    """Build BGE-M3 embeddings for corpus.

    Args:
        corpus_file: Input JSONL file
        output_embeddings: Output .npy file for embeddings
        batch_size: Batch size for encoding
    """
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
    print(f"  - Model: BAAI/bge-m3")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  - Embedding cache: {output_embeddings}")
    print(f"\nEstimated time: ~4-6 hours on H200 GPU for 21M passages...")
    print(f"(You can monitor progress in another terminal)")

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
    corpus_jsonl = raw_dir / "dpr_wikipedia_psgs_w100.jsonl"
    embeddings_file = embeddings_dir / "dpr_wikipedia_bge_m3.npy"

    print("="*70)
    print("META DPR WIKIPEDIA CORPUS SETUP")
    print("="*70)
    print(f"\nThis script will:")
    print(f"1. Download DPR Wikipedia corpus (21M passages) from HuggingFace")
    print(f"2. Convert to JSONL format")
    print(f"3. Generate BGE-M3 embeddings (takes ~4-6 hours on H200 GPU)")
    print(f"\nWhy DPR Wikipedia?")
    print(f"  - Standard corpus for Open-Domain QA (NaturalQuestions, TriviaQA, etc.)")
    print(f"  - 21M passages from Wikipedia (Dec 2018)")
    print(f"  - Used by most ODQA papers for fair comparison")
    print(f"\nData will be saved to:")
    print(f"  - Corpus: {corpus_jsonl}")
    print(f"  - Embeddings: {embeddings_file}")
    print("="*70)

    # Check if already exists
    if corpus_jsonl.exists() and embeddings_file.exists():
        print(f"\n⚠️  Files already exist!")
        print(f"  - {corpus_jsonl}")
        print(f"  - {embeddings_file}")
        print("Skipping download and embedding generation.")
        print("Delete files manually if you want to rebuild.")
        return

    print("\nStarting download and embedding generation...")

    # Step 1: Download and convert corpus
    if not corpus_jsonl.exists():
        print(f"\n{'='*70}")
        print("STEP 1/2: DOWNLOAD AND CONVERT CORPUS")
        print(f"{'='*70}")
        num_docs = download_and_convert_corpus(corpus_jsonl)
    else:
        print(f"\n{'='*70}")
        print("STEP 1/2: USING EXISTING CORPUS")
        print(f"{'='*70}")
        print(f"Corpus file: {corpus_jsonl}")
        with open(corpus_jsonl, 'r') as f:
            num_docs = sum(1 for _ in f)
        print(f"  - Documents: {num_docs:,}")
        print(f"  - Size: {corpus_jsonl.stat().st_size / 1e9:.2f} GB")

    # Step 2: Build embeddings
    if not embeddings_file.exists():
        print(f"\n{'='*70}")
        print("STEP 2/2: BUILD BGE-M3 EMBEDDINGS")
        print(f"{'='*70}")
        build_embeddings(corpus_jsonl, embeddings_file, batch_size=64)
    else:
        print(f"\n{'='*70}")
        print("STEP 2/2: USING EXISTING EMBEDDINGS")
        print(f"{'='*70}")
        print(f"Embeddings file: {embeddings_file}")
        print(f"  - Size: {embeddings_file.stat().st_size / 1e9:.2f} GB")

    # Summary
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE!")
    print(f"{'='*70}")
    print(f"\nCorpus: {corpus_jsonl}")
    print(f"  - Documents: {num_docs:,}")
    if corpus_jsonl.exists():
        print(f"  - Size: {corpus_jsonl.stat().st_size / 1e9:.2f} GB")

    print(f"\nEmbeddings: {embeddings_file}")
    if embeddings_file.exists():
        print(f"  - Size: {embeddings_file.stat().st_size / 1e9:.2f} GB")

    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print(f"\n1. Update your config to use DPR Wikipedia corpus:")
    print(f"   - Corpus: {corpus_jsonl}")
    print(f"   - Embeddings: {embeddings_file}")
    print(f"\n2. Run tests with multiple benchmarks:")
    print(f"   - HotpotQA: python scripts/batch_test_adaptive.py --num-questions 100")
    print(f"   - NaturalQuestions: (add your script)")
    print(f"   - TriviaQA: (add your script)")
    print(f"\n3. This corpus is the standard for Open-Domain QA papers")
    print(f"   - Fair comparison with other methods")
    print(f"   - Works across multiple benchmarks")
    print()


if __name__ == "__main__":
    main()
