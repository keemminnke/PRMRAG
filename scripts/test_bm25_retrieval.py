#!/usr/bin/env python3
"""Test BM25 retrieval to verify it correctly retrieves relevant documents."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval import BM25Retriever
import json


def test_bm25_retrieval():
    """Test BM25 retrieval with example corpus."""

    print("=" * 70)
    print("TEST: BM25 Retrieval Verification")
    print("=" * 70)

    # Load corpus from data file
    print("\n[1] Loading corpus from data file...")
    data_dir = Path(__file__).parent.parent / "data"
    corpus_file = data_dir / "raw" / "example_hotpotqa_corpus.jsonl"

    corpus = []
    with open(corpus_file, 'r') as f:
        for line in f:
            corpus.append(json.loads(line))

    print(f"✓ Loaded {len(corpus)} documents")

    # Show first few documents
    print(f"\n[2] Sample corpus documents:")
    for i, doc in enumerate(corpus[:3]):
        print(f"\n  Doc {i}:")
        print(f"    ID: {doc['id']}")
        print(f"    Title: {doc['title']}")
        print(f"    Text: {doc['text'][:100]}...")

    # Initialize retriever
    print(f"\n[3] Initializing BM25 retriever...")
    retriever = BM25Retriever(corpus=corpus)
    print("✓ Retriever initialized")

    # Test queries
    test_queries = [
        # Query 1: Should retrieve Shirley Temple documents
        {
            "query": "Who played Corliss Archer in Kiss and Tell film?",
            "expected_keywords": ["shirley temple", "kiss and tell", "corliss archer"],
        },
        # Query 2: Should retrieve Chief of Protocol documents
        {
            "query": "Shirley Temple government position Chief of Protocol",
            "expected_keywords": ["chief of protocol", "shirley temple"],
        },
        # Query 3: Should retrieve biographical info
        {
            "query": "Shirley Temple actress diplomat",
            "expected_keywords": ["shirley temple", "actress", "diplomat"],
        },
    ]

    print(f"\n[4] Testing retrieval with {len(test_queries)} queries:")
    print("=" * 70)

    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected_keywords"]

        print(f"\n--- Test Query {i} ---")
        print(f"Query: {query}")
        print(f"Expected keywords: {expected}")

        # Retrieve top-5 documents
        results = retriever.retrieve(query, top_k=5)

        print(f"\nTop {len(results)} retrieved documents:")
        for rank, doc in enumerate(results, 1):
            print(f"\n  Rank {rank} (score: {doc['score']:.4f}):")
            print(f"    Title: {doc['title']}")
            print(f"    Text: {doc['text'][:150]}...")

            # Check if expected keywords are in the document
            doc_text = (doc['title'] + ' ' + doc['text']).lower()
            found_keywords = [kw for kw in expected if kw in doc_text]

            if found_keywords:
                print(f"    ✓ Contains expected keywords: {found_keywords}")
            else:
                print(f"    ⚠ Missing expected keywords")

        # Verify top-1 result contains expected keywords
        top1 = results[0]
        top1_text = (top1['title'] + ' ' + top1['text']).lower()
        matches = [kw for kw in expected if kw in top1_text]

        if matches:
            print(f"\n  ✅ Top-1 result is RELEVANT (contains {matches})")
        else:
            print(f"\n  ❌ Top-1 result may NOT be relevant (missing all expected keywords)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

    print("\n[5] Summary:")
    print("  - BM25 retriever successfully initialized")
    print("  - Retrieval returns ranked documents with scores")
    print("  - Check above to verify relevant documents are ranked highly")


if __name__ == "__main__":
    test_bm25_retrieval()
