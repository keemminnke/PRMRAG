"""BGE-M3 dense retriever for semantic search."""

import warnings
# Suppress XLMRobertaTokenizerFast warning from BGE-M3
warnings.filterwarnings("ignore", message="You're using a XLMRobertaTokenizerFast")

from typing import List, Dict, Any, Optional
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import pickle


class BGERetriever:
    """BGE-M3 dense retriever using semantic embeddings."""

    def __init__(
        self,
        corpus: List[Dict[str, Any]],
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
        embedding_cache_path: Optional[str] = None,
    ):
        """Initialize BGE retriever.

        Args:
            corpus: List of documents, each with 'id', 'title', 'text'
            model_name: HuggingFace model name for BGE
            batch_size: Batch size for encoding
            max_length: Max sequence length
            device: Device to use (None = auto-detect)
            embedding_cache_path: Path to load/save corpus embeddings (None = don't cache)
        """
        self.corpus = corpus
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.embedding_cache_path = embedding_cache_path

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading BGE-M3 model on {self.device}...")
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=(self.device == "cuda")  # Use FP16 on GPU for speed
        )
        print("✓ BGE-M3 model loaded")

        # Load or build embeddings
        if embedding_cache_path and Path(embedding_cache_path).exists():
            print(f"Loading cached embeddings from {embedding_cache_path}...")
            self.doc_embeddings = self._load_embeddings(embedding_cache_path)
            print(f"✓ Loaded {len(self.doc_embeddings)} cached embeddings")
        else:
            print(f"Building dense embeddings for {len(corpus)} documents...")
            self.doc_embeddings = self._encode_corpus()
            print("✓ Dense index built!")

            # Save embeddings if cache path provided
            if embedding_cache_path:
                self._save_embeddings(embedding_cache_path)
                print(f"✓ Saved embeddings to {embedding_cache_path}")

    def _encode_corpus(self) -> np.ndarray:
        """Encode all corpus documents into embeddings.

        Returns:
            Array of shape (num_docs, embedding_dim)
        """
        # Prepare texts (title + text)
        texts = [
            f"{doc['title']} {doc['text']}"
            for doc in self.corpus
        ]

        # Encode in batches
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding corpus"):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                max_length=self.max_length,
                batch_size=len(batch),
            )['dense_vecs']
            all_embeddings.append(embeddings)

        # Concatenate all batches
        embeddings = np.vstack(all_embeddings)
        return embeddings

    def _save_embeddings(self, cache_path: str):
        """Save corpus embeddings to disk.

        Args:
            cache_path: Path to save embeddings
        """
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as numpy array (more efficient than pickle)
        np.save(cache_path, self.doc_embeddings)

    def _load_embeddings(self, cache_path: str) -> np.ndarray:
        """Load corpus embeddings from disk.

        Args:
            cache_path: Path to load embeddings from

        Returns:
            Loaded embeddings array
        """
        return np.load(cache_path)

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
        # Encode query
        query_embedding = self.model.encode(
            [query],
            max_length=self.max_length,
            batch_size=1,
        )['dense_vecs'][0]

        # Compute cosine similarity with all documents
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Normalize doc embeddings
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        doc_embeddings_norm = self.doc_embeddings / doc_norms

        # Compute similarity scores
        scores = np.dot(doc_embeddings_norm, query_norm)

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
        # Encode all queries at once
        query_embeddings = self.model.encode(
            queries,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )['dense_vecs']

        # Normalize
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings_norm = query_embeddings / query_norms

        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        doc_embeddings_norm = self.doc_embeddings / doc_norms

        # Compute similarity matrix (queries x docs)
        scores_matrix = np.dot(query_embeddings_norm, doc_embeddings_norm.T)

        # Get top-k for each query
        all_results = []
        for i, query in enumerate(queries):
            scores = scores_matrix[i]
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
            all_results.append(results)

        return all_results
