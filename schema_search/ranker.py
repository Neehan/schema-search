from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict
from abc import ABC, abstractmethod
import logging

import numpy as np
from rank_bm25 import BM25Okapi

from schema_search.chunker import Chunk
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class BaseRanker(ABC):
    """Base class for ranking implementations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunks: Optional[List[Chunk]] = None

    @abstractmethod
    def build(self, chunks: List[Chunk]):
        """Build/initialize the ranker with chunks."""
        pass

    @abstractmethod
    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        """
        Rank chunks based on query.

        Returns:
            List of (chunk_idx, combined_score, embedding_score, auxiliary_score)
        """
        pass

    def get_top_tables_from_chunks(
        self, ranked_chunks: List[Tuple[int, float, float, float]], top_k: int
    ) -> Dict[str, List[int]]:
        """Extract top tables from ranked chunks."""
        if self.chunks is None:
            raise RuntimeError("Chunks not initialized. Call build() first")

        table_to_chunk_indices = defaultdict(list)

        for chunk_idx, score, emb_score, aux_score in ranked_chunks:
            chunk = self.chunks[chunk_idx]
            table_to_chunk_indices[chunk.table_name].append(chunk_idx)

        table_scores = {}
        for table_name, chunk_indices in table_to_chunk_indices.items():
            max_score = max(ranked_chunks[idx][1] for idx in chunk_indices)
            table_scores[table_name] = max_score

        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        result = {}
        for table_name, score in top_tables:
            result[table_name] = table_to_chunk_indices[table_name]

        return result


class BM25Ranker(BaseRanker):
    """Hybrid ranker using BM25 + embeddings."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.bm25 = None

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        tokens = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokens)
        logger.debug(f"Built BM25 index with {len(chunks)} chunks")

    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first")

        embedding_scores = (embeddings @ query_embedding.T).flatten()
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)

        embedding_weight = self.config["search"]["embedding_weight"]
        bm25_weight = self.config["search"]["bm25_weight"]

        combined_scores = (
            embedding_weight * embedding_scores + bm25_weight * bm25_scores_norm
        )

        ranked_indices = combined_scores.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            results.append(
                (
                    int(idx),
                    float(combined_scores[idx]),
                    float(embedding_scores[idx]),
                    float(bm25_scores_norm[idx]),
                )
            )

        return results


class CrossEncoderReranker(BaseRanker):
    """Ranker using CrossEncoder for reranking."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self._load_model()

    def _load_model(self):
        model_name = self.config["search"].get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded CrossEncoder: {model_name}")

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        logger.debug(f"Initialized CrossEncoder reranker with {len(chunks)} chunks")

    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        if self.model is None:
            raise RuntimeError("CrossEncoder model not loaded")
        if self.chunks is None:
            raise RuntimeError("Chunks not initialized. Call build() first")

        # First pass: embedding similarity to narrow down candidates
        embedding_scores = (embeddings @ query_embedding.T).flatten()
        top_k = self.config["search"]["initial_top_k"]
        top_indices = embedding_scores.argsort()[::-1][:top_k]

        # Second pass: CrossEncoder reranking
        pairs = [(query, self.chunks[idx].content) for idx in top_indices]
        rerank_scores = self.model.predict(pairs)

        # Combine: use rerank scores for top_k, embedding scores for rest
        final_scores = np.zeros(len(embeddings))
        final_scores[top_indices] = rerank_scores

        # For chunks not reranked, use normalized embedding scores (lower weight)
        remaining_indices = np.setdiff1d(np.arange(len(embeddings)), top_indices)
        if len(remaining_indices) > 0:
            final_scores[remaining_indices] = embedding_scores[remaining_indices] * 0.5

        ranked_indices = final_scores.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            results.append(
                (
                    int(idx),
                    float(final_scores[idx]),
                    float(embedding_scores[idx]),
                    (
                        float(rerank_scores[np.where(top_indices == idx)[0][0]])
                        if idx in top_indices
                        else 0.0
                    ),
                )
            )

        return results


# Factory function for creating rankers
def create_ranker(config: Dict[str, Any]) -> BaseRanker:
    """Create a ranker based on config. Uses CrossEncoder if reranker_model is set."""
    reranker_model = config["search"].get("reranker_model")

    if reranker_model:
        return CrossEncoderReranker(config)
    else:
        return BM25Ranker(config)


# Backward compatibility alias
Ranker = BM25Ranker
