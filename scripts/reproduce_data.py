#!/usr/bin/env python3
"""Reproduce data environment: Download datasets and build/cache embeddings.

This script allows you to replicate the data setup on a new server.
It performs the following steps:
1. Downloads KILT knowledge source (35GB).
2. Downloads and prepares HotpotQA (Train/Dev).
3. Downloads and prepares MuSiQue (Train/Dev).
4. (Optional) Builds BM25 Index for KILT.
5. (Optional) Builds BGE-M3 Embeddings for KILT (Computationally intensive).

Usage:
    python scripts/reproduce_data.py --build-index --build-embeddings
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bm25_retriever import BM25Retriever

def run_command(cmd: List[str], desc: str):
    print(f"\n[Running] {desc}")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def download_kilt(data_dir: Path):
    kilt_dir = data_dir / "kilt"
    kilt_file = kilt_dir / "kilt_knowledgesource.json"
    url = "http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json"

    if kilt_file.exists():
        print(f"\n✓ KILT corpus found at {kilt_file}")
        return

    print(f"\n[Download] KILT Knowledge Source (35GB)...")
    kilt_dir.mkdir(parents=True, exist_ok=True)
    # Use wget with continue flag
    try:
        subprocess.run(["wget", "-c", url, "-O", str(kilt_file)], check=True)
    except FileNotFoundError:
        print("Error: 'wget' not found. Please install wget or download manually.")
        sys.exit(1)

def get_truncated_text(doc_text_list, max_paragraphs=10, max_chars=1000):
    """Smart truncation: paragraph-based with character safety net."""
    if not doc_text_list:
        return ""
    # 1. Take first N paragraphs
    selected_paragraphs = doc_text_list[:max_paragraphs]
    # 2. Join them
    joined_text = ' '.join(selected_paragraphs)
    # 3. Safety net
    if len(joined_text) > max_chars:
        joined_text = joined_text[:max_chars] + "..."
    return joined_text

def load_kilt_corpus(corpus_file: Path, limit: int = None) -> List[Dict[str, Any]]:
    print(f"Loading KILT corpus from {corpus_file}...")
    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            # KILT format: text is a list of paragraphs
            if isinstance(doc['text'], list):
                text = get_truncated_text(doc['text'], max_paragraphs=2, max_chars=1200)
            else:
                text = doc['text'][:1200] + ("..." if len(doc['text']) > 1200 else "")
            
            corpus.append({
                'id': doc['_id'],
                'title': doc['wikipedia_title'],
                'text': text,
            })
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i+1:,} documents...", end='\r')
    print(f"\n✓ Loaded {len(corpus):,} documents")
    return corpus

def main():
    parser = argparse.ArgumentParser(description="Reproduce data and embeddings.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data root directory")
    parser.add_argument("--build-index", action="store_true", help="Build BM25 index")
    parser.add_argument("--build-embeddings", action="store_true", help="Build BGE-M3 embeddings (Slow!)")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus size (for testing)")
    args = parser.parse_args()

    # 1. Download KILT
    download_kilt(args.data_dir)

    # 2. Prepare Questions
    print("\n[Questions] Preparing Question Datasets...")
    scripts_dir = Path(__file__).parent
    
    # HotpotQA
    run_command([sys.executable, str(scripts_dir / "prepare_hotpotqa_questions.py")], "Prepare HotpotQA")
    
    # MuSiQue
    run_command([sys.executable, str(scripts_dir / "prepare_musique_questions.py")], "Prepare MuSiQue")

    # 3. Build Indexes (if requested)
    if args.build_index or args.build_embeddings:
        corpus_path = args.data_dir / "kilt" / "kilt_knowledgesource.json"
        corpus = load_kilt_corpus(corpus_path, limit=args.limit)

        if args.build_index:
            index_path = args.data_dir / "indexes" / "kilt_wikipedia_bm25.pkl"
            print(f"\n[BM25] Building index at {index_path}...")
            BM25Retriever(corpus=corpus, index_cache_path=str(index_path))
        
        if args.build_embeddings:
            embed_path = args.data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"
            print(f"\n[BGE-M3] Building embeddings at {embed_path}...")
            print("Warning: This process requires a GPU and may take a long time.")
            BGERetriever(corpus=corpus, embedding_cache_path=str(embed_path), batch_size=64)

    print("\n✓ Data reproduction setup complete!")

if __name__ == "__main__":
    main()
