#!/usr/bin/env python3
"""
Setup BeIR HotpotQA corpus with BGE-M3 embeddings.

This script:
1. Downloads BeIR/hotpotqa corpus from HuggingFace
2. Saves corpus to JSONL format
3. Generates BGE-M3 embeddings for the entire corpus

Questions (queries) are already available from setup_fullwiki_corpus.py:
- data/raw/questions/hotpotqa_train.jsonl
- data/raw/questions/hotpotqa_validation.jsonl
- data/raw/questions/hotpotqa_test.jsonl
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from FlagEmbedding import BGEM3FlagModel


def setup_beir_corpus():
    """Download BeIR HotpotQA corpus and generate BGE-M3 embeddings."""

    # Paths
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    embeddings_dir = data_dir / "embeddings"

    raw_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Files
    corpus_file = raw_dir / "beir_hotpotqa_corpus.jsonl"
    embeddings_file = embeddings_dir / "beir_hotpotqa_bge_m3.npy"

    print("="*70)
    print("BEIR HOTPOTQA CORPUS SETUP")
    print("="*70)
    print(f"\nThis script will:")
    print(f"1. Download BeIR/hotpotqa corpus from HuggingFace")
    print(f"2. Save corpus to JSONL format")
    print(f"3. Generate BGE-M3 embeddings")
    print(f"\nData will be saved to:")
    print(f"  - Corpus: {corpus_file}")
    print(f"  - Embeddings: {embeddings_file}")
    print(f"\nQuestions already available:")
    print(f"  - data/raw/questions/hotpotqa_train.jsonl")
    print(f"  - data/raw/questions/hotpotqa_validation.jsonl")
    print(f"  - data/raw/questions/hotpotqa_test.jsonl")
    print("="*70)

    # Check if already exists
    if corpus_file.exists() and embeddings_file.exists():
        print(f"\n⚠️  Files already exist!")
        print(f"  - {corpus_file}")
        print(f"  - {embeddings_file}")
        response = input("\nDo you want to rebuild? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    input("\nPress Enter to start, or Ctrl+C to cancel...")

    # Step 1 & 2: Download and save corpus (skip if exists)
    if not corpus_file.exists():
        print(f"\n[1/3] Downloading BeIR/hotpotqa corpus from HuggingFace...")
        corpus = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
        print(f"  - Corpus size: {len(corpus):,} documents")

        print(f"\n[2/3] Saving corpus to JSONL...")
        with open(corpus_file, 'w', encoding='utf-8') as f:
            for doc in tqdm(corpus, desc="Writing corpus"):
                f.write(json.dumps({
                    'id': doc['_id'],
                    'title': doc['title'],
                    'text': doc['text']
                }, ensure_ascii=False) + '\n')

        print(f"  ✅ Saved {len(corpus):,} documents to {corpus_file}")
        print(f"  - File size: {corpus_file.stat().st_size / 1e6:.2f} MB")
    else:
        print(f"\n[1-2/3] Corpus already exists, skipping download...")
        print(f"  - Using: {corpus_file}")
        print(f"  - Size: {corpus_file.stat().st_size / 1e6:.2f} MB")

    # Step 3: Load corpus from JSONL and generate embeddings
    print(f"\n[3/3] Generating BGE-M3 embeddings...")
    print(f"  - Loading corpus from JSONL...")

    corpus_data = []
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            corpus_data.append(json.loads(line))

    print(f"  - Loaded {len(corpus_data):,} documents")
    print(f"  - Model: BAAI/bge-m3")
    print(f"  - This may take several hours depending on corpus size and GPU...")

    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    embeddings = []
    batch_size = 256  # Increased for H200

    for i in tqdm(range(0, len(corpus_data), batch_size), desc="Encoding batches"):
        batch = corpus_data[i:i+batch_size]
        texts = [f"{doc['title']} {doc['text']}" for doc in batch]
        batch_embs = model.encode(
            texts,
            max_length=512
        )['dense_vecs']
        embeddings.append(batch_embs)

    # Combine all batches
    embeddings = np.vstack(embeddings)

    # Save embeddings
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)

    print(f"  ✅ Saved embeddings: {embeddings.shape}")
    print(f"  - File size: {embeddings_file.stat().st_size / 1e9:.2f} GB")

    # Summary
    print(f"\n{'='*70}")
    print("✅ SETUP COMPLETE!")
    print(f"{'='*70}")
    print(f"\nCorpus: {corpus_file}")
    print(f"  - Documents: {len(corpus_data):,}")
    print(f"  - Size: {corpus_file.stat().st_size / 1e6:.2f} MB")
    print(f"\nEmbeddings: {embeddings_file}")
    print(f"  - Shape: {embeddings.shape}")
    print(f"  - Size: {embeddings_file.stat().st_size / 1e9:.2f} GB")
    print(f"\nQuestions (already available):")
    print(f"  - data/raw/questions/hotpotqa_train.jsonl")
    print(f"  - data/raw/questions/hotpotqa_validation.jsonl")
    print(f"  - data/raw/questions/hotpotqa_test.jsonl")
    print(f"\nYou can now run batch tests:")
    print(f"  python scripts/batch_test_adaptive.py --num-questions 10")
    print()


if __name__ == "__main__":
    setup_beir_corpus()
