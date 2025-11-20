#!/usr/bin/env python3
"""Download and prepare HotpotQA Full Wiki corpus (5.23M paragraphs) with BGE-M3 embeddings.

This script:
1. Downloads the full Wikipedia corpus used in HotpotQA
2. Converts to JSONL format compatible with BGERetriever
3. Generates BGE-M3 embeddings (takes ~2-4 hours on GPU)
4. Saves embeddings cache for fast retrieval
"""

import sys
import json
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval import BGERetriever


def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    print(f"Downloading from {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc=output_path.name
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"✓ Downloaded to {output_path}")


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract zip file."""
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"✓ Extracted to {extract_to}")


def convert_to_jsonl(input_file: Path, output_file: Path):
    """Convert HotpotQA wiki format to JSONL format.

    Input format: JSON with article_title -> [paragraph_text, ...]
    Output format: JSONL with {id, title, text} per line
    """
    print(f"Converting {input_file.name} to JSONL...")

    with open(input_file, 'r', encoding='utf-8') as f:
        wiki_data = json.load(f)

    doc_id = 0
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for title, paragraphs in tqdm(wiki_data.items(), desc="Converting"):
            for paragraph in paragraphs:
                # Skip empty paragraphs
                if not paragraph.strip():
                    continue

                doc = {
                    'id': f'wiki_{doc_id}',
                    'title': title,
                    'text': paragraph.strip()
                }
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
                doc_id += 1

    print(f"✓ Converted {doc_id:,} documents to {output_file}")
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
    zip_file = raw_dir / "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
    extracted_file = raw_dir / "enwiki-20171001-pages-meta-current-withlinks-abstracts"
    corpus_jsonl = raw_dir / "hotpotqa_fullwiki_corpus.jsonl"
    embeddings_file = embeddings_dir / "hotpotqa_fullwiki_bge_m3.npy"

    print("="*70)
    print("HOTPOTQA FULL WIKI CORPUS SETUP")
    print("="*70)
    print(f"\nThis script will:")
    print(f"1. Download 5.23M Wikipedia abstracts (~2GB)")
    print(f"2. Convert to JSONL format")
    print(f"3. Generate BGE-M3 embeddings (~50GB, takes 2-4 hours)")
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

    # Step 1: Download
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"

    if not extracted_file.exists():
        if not zip_file.exists():
            print(f"\n[1/3] Downloading corpus...")
            download_file(url, zip_file)
        else:
            print(f"\n[1/3] Using cached download: {zip_file}")

        # Extract
        print(f"\n[2/3] Extracting...")
        import tarfile
        with tarfile.open(zip_file, 'r:bz2') as tar:
            tar.extractall(raw_dir)
        print(f"✓ Extracted")
    else:
        print(f"\n[1-2/3] Using existing extracted file: {extracted_file}")

    # Step 2: Convert to JSONL
    if not corpus_jsonl.exists():
        print(f"\n[3/3] Converting to JSONL...")
        num_docs = convert_to_jsonl(extracted_file, corpus_jsonl)
    else:
        print(f"\n[3/3] Using existing JSONL: {corpus_jsonl}")
        with open(corpus_jsonl, 'r') as f:
            num_docs = sum(1 for _ in f)
        print(f"  - Documents: {num_docs:,}")

    # Step 3: Build embeddings
    if not embeddings_file.exists():
        print(f"\n[4/3] Building BGE-M3 embeddings...")
        build_embeddings(corpus_jsonl, embeddings_file, batch_size=64)
    else:
        print(f"\n[4/3] Using existing embeddings: {embeddings_file}")

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
    print(f"\nYou can now use Full Wiki setting with:")
    print(f"  python scripts/batch_test_adaptive.py --use-fullwiki")
    print()


if __name__ == "__main__":
    main()
