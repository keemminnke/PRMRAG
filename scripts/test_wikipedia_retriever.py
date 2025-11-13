#!/usr/bin/env python3
"""
Test Wikipedia retriever.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.retrieval.wikipedia_retriever import WikipediaRetriever


def main():
    print("Testing Wikipedia Retriever\n")
    print("=" * 60)

    # Initialize
    retriever = WikipediaRetriever(
        lang='en',
        user_agent='PRMRAG/1.0',
        extract_sentences=3
    )

    # Test queries
    queries = [
        "Paris France capital",
        "Harry Potter author",
        "largest planet solar system",
        "Python programming language",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        results = retriever.retrieve(query, top_k=3)

        if not results:
            print("  No results found")
            continue

        for i, doc in enumerate(results, 1):
            print(f"\n  [{i}] {doc['title']}")
            print(f"      Score: {doc['score']:.2f}")
            print(f"      Text: {doc['text'][:150]}...")
            print(f"      URL: {doc['url']}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()
