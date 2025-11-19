"""Retrieval module for RAG."""

from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict


class BaseRetriever:
    """Base class for all retrievers."""

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dicts with keys: doc_id, title, text, score
        """
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    """BM25-based retriever for RAG intervention."""

    def __init__(self, corpus: List[Dict[str, Any]], tokenize_fn=None):
        """Initialize BM25 retriever.

        Args:
            corpus: List of documents, each with 'id', 'title', 'text'
            tokenize_fn: Optional tokenization function
        """
        self.corpus = corpus
        self.tokenize_fn = tokenize_fn or self._default_tokenize

        # Build index
        self.doc_ids = [doc['id'] for doc in corpus]
        self.tokenized_corpus = [
            self.tokenize_fn(doc['title'] + ' ' + doc['text'])
            for doc in corpus
        ]

        print(f"Building BM25 index over {len(corpus)} documents...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("BM25 index built!")

    def _default_tokenize(self, text: str) -> List[str]:
        """Default tokenization (simple whitespace split + lowercase)."""
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        tokenized_query = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_k_indices:
            doc = self.corpus[idx]
            results.append({
                'doc_id': doc['id'],
                'title': doc['title'],
                'text': doc['text'],
                'score': float(scores[idx]),
            })

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries.

        Args:
            queries: List of search queries
            top_k: Number of documents per query

        Returns:
            List of retrieval results
        """
        return [self.retrieve(q, top_k) for q in queries]


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


class HybridRetriever:
    """Hybrid retriever combining multiple retrieval methods (future extension)."""

    def __init__(self, retrievers: List[Any], weights: List[float] = None):
        """Initialize hybrid retriever.

        Args:
            retrievers: List of retriever instances
            weights: Weights for each retriever (optional)
        """
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using multiple retrievers and combine scores."""
        all_results = defaultdict(lambda: {'doc': None, 'score': 0.0})

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, top_k=top_k * 2)

            for result in results:
                doc_id = result['doc_id']
                if all_results[doc_id]['doc'] is None:
                    all_results[doc_id]['doc'] = result
                all_results[doc_id]['score'] += weight * result['score']

        # Sort by combined score
        ranked = sorted(
            all_results.values(),
            key=lambda x: x['score'],
            reverse=True
        )

        return [item['doc'] for item in ranked[:top_k]]


# Import Wikipedia retriever
from .wikipedia_retriever import WikipediaRetriever, create_retriever

# Import BGE retriever
from .bge_retriever import BGERetriever

__all__ = [
    'BaseRetriever',
    'BM25Retriever',
    'BGERetriever',
    'WikipediaRetriever',
    'load_hotpotqa_corpus',
    'HybridRetriever',
    'create_retriever',
]
