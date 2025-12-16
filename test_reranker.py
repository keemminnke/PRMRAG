#!/usr/bin/env python3
"""Test BGE Reranker with actual retrieval examples."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bm25_retriever import BM25Retriever
from prmrag.retrieval.hybrid_retriever import HybridRetriever
from prmrag.retrieval.bge_reranker import BGEReranker


def load_kilt_corpus_truncated(corpus_file: Path, limit: int = None):
    """Load KILT corpus with smart truncation"""
    def get_truncated_text(doc_text_list, max_paragraphs=2, max_chars=1200):
        if not doc_text_list:
            return ""
        selected_paragraphs = doc_text_list[:max_paragraphs]
        joined_text = ' '.join(selected_paragraphs)
        if len(joined_text) > max_chars:
            joined_text = joined_text[:max_chars] + "..."
        return joined_text

    print(f"Loading KILT corpus from {corpus_file}...")
    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
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
                print(f"  Loaded {i+1:,} documents...")
    print(f"✓ Loaded {len(corpus):,} documents")
    return corpus


def main():
    print("="*80)
    print("BGE RERANKER TEST")
    print("="*80)
    print()

    # Test queries (from failed cases)
    test_cases = [
        {
            'query': "Which magazine was started first Arthur's Magazine or First for Women?",
            'expected': "Arthur's Magazine",
        },
        {
            'query': "The Great Outdoors 1988 film cast four-time Academy Award nominee",
            'expected': "Dan Aykroyd",
        },
    ]

    # Load corpus
    data_dir = Path("data")
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    corpus = load_kilt_corpus_truncated(corpus_file)

    # Initialize retrievers
    print("\n[1] Initializing retrievers...")
    embedding_cache = data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"
    bge_retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=str(embedding_cache)
    )

    bm25_cache = data_dir / "indexes" / "kilt_wikipedia_bm25.pkl"
    bm25_retriever = BM25Retriever(
        corpus=corpus,
        index_cache_path=str(bm25_cache)
    )

    # Test WITHOUT reranker
    print("\n[2] Testing WITHOUT reranker...")
    hybrid_no_rerank = HybridRetriever(
        bm25_retriever=bm25_retriever,
        bge_retriever=bge_retriever,
        fusion_method="rrf",
        k_sparse=50,
        k_dense=50,
    )

    # Test WITH reranker
    print("\n[3] Initializing reranker...")
    reranker = BGEReranker(device="cuda")

    print("\n[4] Testing WITH reranker...")
    hybrid_with_rerank = HybridRetriever(
        bm25_retriever=bm25_retriever,
        bge_retriever=bge_retriever,
        fusion_method="rrf",
        k_sparse=50,
        k_dense=50,
        reranker=reranker,
        rerank_top_n=20,  # Rerank top 20 candidates
    )

    # Run tests
    for idx, test in enumerate(test_cases, 1):
        query = test['query']
        expected = test['expected']

        print(f"\n{'='*80}")
        print(f"Test {idx}: {query[:60]}...")
        print(f"Expected: {expected}")
        print(f"{'='*80}")

        # WITHOUT reranker
        print("\n--- WITHOUT Reranker ---")
        results_no_rerank = hybrid_no_rerank.retrieve(query, top_k=5)
        for i, doc in enumerate(results_no_rerank, 1):
            print(f"[{i}] {doc['title'][:60]}")
            print(f"    Score: {doc['score']:.4f}")
            if expected.lower() in doc['title'].lower():
                print(f"    ✅ FOUND EXPECTED!")

        # WITH reranker
        print("\n--- WITH Reranker ---")
        results_with_rerank = hybrid_with_rerank.retrieve(query, top_k=5)
        for i, doc in enumerate(results_with_rerank, 1):
            print(f"[{i}] {doc['title'][:60]}")
            print(f"    Fusion Score: {doc['score']:.4f}")
            if 'rerank_score' in doc:
                print(f"    Rerank Score: {doc['rerank_score']:.4f}")
            if expected.lower() in doc['title'].lower():
                print(f"    ✅ FOUND EXPECTED!")

        # Compare
        print("\n--- Comparison ---")
        no_rerank_titles = [d['title'] for d in results_no_rerank]
        with_rerank_titles = [d['title'] for d in results_with_rerank]

        # Check if expected is in top-5
        expected_in_no_rerank = any(expected.lower() in t.lower() for t in no_rerank_titles)
        expected_in_with_rerank = any(expected.lower() in t.lower() for t in with_rerank_titles)

        print(f"Expected in top-5 (NO reranker):   {'✅' if expected_in_no_rerank else '❌'}")
        print(f"Expected in top-5 (WITH reranker): {'✅' if expected_in_with_rerank else '❌'}")

        if not expected_in_no_rerank and expected_in_with_rerank:
            print("🎯 RERANKER IMPROVED RESULTS!")
        elif expected_in_no_rerank and not expected_in_with_rerank:
            print("⚠️  RERANKER MADE IT WORSE")
        elif expected_in_no_rerank and expected_in_with_rerank:
            # Check if ranking improved
            no_rerank_rank = next((i for i, t in enumerate(no_rerank_titles) if expected.lower() in t.lower()), None)
            with_rerank_rank = next((i for i, t in enumerate(with_rerank_titles) if expected.lower() in t.lower()), None)
            if with_rerank_rank < no_rerank_rank:
                print(f"📈 RANKING IMPROVED: #{no_rerank_rank+1} → #{with_rerank_rank+1}")
            elif with_rerank_rank > no_rerank_rank:
                print(f"📉 RANKING DEGRADED: #{no_rerank_rank+1} → #{with_rerank_rank+1}")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
