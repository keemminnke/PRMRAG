#!/usr/bin/env python3
"""Test RAG components and profile timing."""

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

    # [1] Load corpus (limit for quick test)
    print("=" * 60)
    print("[1/4] Loading KILT corpus (first 100k docs for quick test)...")
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"

    t0 = time.time()
    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100000:  # Limit for quick test
                break
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
    print(f"  Loaded {len(corpus):,} docs in {time.time()-t0:.1f}s")

    # [2] Initialize BGE Retriever
    print("\n" + "=" * 60)
    print("[2/4] Initializing BGE-M3 Retriever...")

    from prmrag.retrieval.bge_retriever import BGERetriever

    # Note: Won't use cache for 100k - just test model loading
    t0 = time.time()
    bge_retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=None,  # Skip cache for test
    )
    print(f"  Initialized in {time.time()-t0:.1f}s")

    # [3] Test single retrieval
    print("\n" + "=" * 60)
    print("[3/4] Testing single query retrieval...")

    test_query = "Who was the president of United States in 2020?"

    t0 = time.time()
    results = bge_retriever.retrieve(test_query, top_k=5)
    t_single = time.time() - t0

    print(f"  Single retrieval time: {t_single*1000:.1f}ms")
    print(f"  Results:")
    for i, r in enumerate(results):
        print(f"    [{i+1}] {r['title']}: {r['text'][:80]}... (score: {r['score']:.3f})")

    # [4] Test batch retrieval
    print("\n" + "=" * 60)
    print("[4/4] Testing batch retrieval (10 queries)...")

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

    t0 = time.time()
    batch_results = bge_retriever.batch_retrieve(test_queries, top_k=5)
    t_batch = time.time() - t0

    print(f"  Batch retrieval time: {t_batch*1000:.1f}ms ({t_batch*1000/len(test_queries):.1f}ms per query)")
    print(f"  First 3 results for first query:")
    for i, r in enumerate(batch_results[0][:3]):
        print(f"    [{i+1}] {r['title']}: {r['text'][:60]}...")

    # [5] Test Reranker
    print("\n" + "=" * 60)
    print("[5/5] Testing BGE Reranker...")

    from prmrag.retrieval.bge_reranker import BGEReranker

    t0 = time.time()
    reranker = BGEReranker(device="cuda", batch_size=64)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Rerank single query
    t0 = time.time()
    reranked = reranker.rerank(test_query, results, top_k=3)
    t_rerank = time.time() - t0

    print(f"  Single rerank time: {t_rerank*1000:.1f}ms")
    print(f"  Reranked top 3:")
    for i, r in enumerate(reranked):
        print(f"    [{i+1}] {r['title']} (rerank_score: {r['rerank_score']:.3f})")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Single retrieval (BGE): {t_single*1000:.1f}ms")
    print(f"  Batch retrieval (BGE, 10 queries): {t_batch*1000:.1f}ms total, {t_batch*1000/len(test_queries):.1f}ms/query")
    print(f"  Single rerank (5 docs): {t_rerank*1000:.1f}ms")
    print("=" * 60)


if __name__ == '__main__':
    main()
