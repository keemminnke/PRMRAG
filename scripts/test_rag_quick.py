#!/usr/bin/env python3
"""Quick test: Load cached embeddings and test retrieval timing."""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def get_truncated_text(doc_text_list, max_chars=1200):
    if not doc_text_list:
        return ""
    joined_text = ' '.join(doc_text_list)
    if len(joined_text) > max_chars:
        return joined_text[:max_chars] + " [TRUNCATED]"
    return joined_text


def main():
    data_dir = Path(__file__).parent.parent / "data"
    embedding_cache = data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"

    # [1] Load corpus
    print("=" * 60)
    print("[1/4] Loading KILT corpus...")
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"

    t0 = time.time()
    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            if isinstance(doc['text'], list):
                text = get_truncated_text(doc['text'], max_chars=1200)
            else:
                text = doc['text'][:1200]
            corpus.append({
                'id': doc['_id'],
                'title': doc['wikipedia_title'],
                'text': text,
            })
            if (i + 1) % 1000000 == 0:
                print(f"  Loaded {i+1:,} docs...")
    print(f"  Total: {len(corpus):,} docs in {time.time()-t0:.1f}s")

    # [2] Initialize BGE Retriever with cached embeddings
    print("\n" + "=" * 60)
    print(f"[2/4] Loading BGE-M3 + cached embeddings from {embedding_cache}...")

    from prmrag.retrieval.bge_retriever import BGERetriever

    t0 = time.time()
    bge_retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=str(embedding_cache),
    )
    print(f"  Initialized in {time.time()-t0:.1f}s")
    print(f"  Embedding shape: {bge_retriever.doc_embeddings.shape}")

    # [3] Test single retrieval
    print("\n" + "=" * 60)
    print("[3/4] Testing single query retrieval (5.9M docs)...")

    test_query = "Who was the president of United States in 2020?"

    # Warmup
    _ = bge_retriever.retrieve(test_query, top_k=5)

    # Measure
    t0 = time.time()
    results = bge_retriever.retrieve(test_query, top_k=50)
    t_single = time.time() - t0

    print(f"  Single retrieval time: {t_single*1000:.1f}ms")
    print(f"  Top 3 results:")
    for i, r in enumerate(results[:3]):
        print(f"    [{i+1}] {r['title']}: {r['text'][:60]}... (score: {r['score']:.3f})")

    # [4] Test batch retrieval
    print("\n" + "=" * 60)
    print("[4/4] Testing batch retrieval (10 queries x 50 candidates)...")

    test_queries = [
        "Who won the 2020 presidential election?",
        "What is the capital of France?",
        "When was the Eiffel Tower built?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "Who discovered penicillin?",
        "What is the largest ocean?",
        "Who painted the Mona Lisa?",
        "What is the chemical formula for water?",
        "Who invented the telephone?",
    ]

    # Warmup
    _ = bge_retriever.batch_retrieve(test_queries[:2], top_k=50)

    # Measure
    t0 = time.time()
    batch_results = bge_retriever.batch_retrieve(test_queries, top_k=50)
    t_batch = time.time() - t0

    print(f"  Batch retrieval (10 queries): {t_batch*1000:.1f}ms total, {t_batch*1000/len(test_queries):.1f}ms/query")

    # [5] Test Reranker
    print("\n" + "=" * 60)
    print("[5/5] Testing BGE Reranker...")

    from prmrag.retrieval.bge_reranker import BGEReranker

    t0 = time.time()
    reranker = BGEReranker(device="cuda", batch_size=64)
    print(f"  Reranker loaded in {time.time()-t0:.1f}s")

    # Single rerank (50 docs)
    t0 = time.time()
    reranked = reranker.rerank(test_query, results, top_k=5)
    t_rerank_single = time.time() - t0

    print(f"  Single rerank (50 docs -> 5): {t_rerank_single*1000:.1f}ms")
    print(f"  Reranked top 3:")
    for i, r in enumerate(reranked[:3]):
        print(f"    [{i+1}] {r['title']} (rerank_score: {r['rerank_score']:.3f})")

    # Batch rerank
    t0 = time.time()
    batch_reranked = reranker.batch_rerank(test_queries, batch_results, top_k=5)
    t_rerank_batch = time.time() - t0

    print(f"  Batch rerank (10 queries x 50 docs): {t_rerank_batch*1000:.1f}ms total, {t_rerank_batch*1000/len(test_queries):.1f}ms/query")

    # Summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY (5.9M docs corpus)")
    print("=" * 60)
    print(f"  BGE retrieval (single, 50 candidates): {t_single*1000:.1f}ms")
    print(f"  BGE retrieval (batch 10, 50 candidates/q): {t_batch*1000:.1f}ms ({t_batch*1000/10:.1f}ms/query)")
    print(f"  Rerank (single, 50 -> 5): {t_rerank_single*1000:.1f}ms")
    print(f"  Rerank (batch 10, 50 -> 5): {t_rerank_batch*1000:.1f}ms ({t_rerank_batch*1000/10:.1f}ms/query)")
    print(f"\n  Full RAG step estimate (retrieval + rerank):")
    print(f"    Single: ~{(t_single + t_rerank_single)*1000:.0f}ms")
    print(f"    Batch 10: ~{(t_batch + t_rerank_batch)*1000:.0f}ms ({(t_batch + t_rerank_batch)*1000/10:.0f}ms/query)")
    print("=" * 60)


if __name__ == '__main__':
    main()
