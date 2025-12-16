"""Retrieval module for RAG."""

from typing import List, Dict, Any

# Import retrievers
from .bge_retriever import BGERetriever
from .bm25_retriever import BM25Retriever
from .hybrid_retriever import HybridRetriever
from .wikipedia_retriever import WikipediaRetriever, create_retriever
from .bge_reranker import BGEReranker


def load_hotpotqa_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load HotpotQA corpus for retrieval.

    Args:
        corpus_path: Path to corpus file (JSONL)

    Returns:
        List of documents
    """
    import jsonlines

    corpus = []
    with jsonlines.open(corpus_path) as reader:
        for obj in reader:
            corpus.append({
                'id': obj.get('id', f"doc_{len(corpus)}"),
                'title': obj.get('title', ''),
                'text': obj.get('text', ''),
            })

    return corpus


__all__ = [
    'BGERetriever',
    'BM25Retriever',
    'HybridRetriever',
    'WikipediaRetriever',
    'BGEReranker',
    'load_hotpotqa_corpus',
    'create_retriever',
]
