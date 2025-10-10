from typing import List, Tuple
import logging

import numpy as np
from rank_bm25 import BM25Okapi

from schema_search.chunkers import Chunk
from schema_search.rankers.base import BaseRanker

logger = logging.getLogger(__name__)


class BM25Ranker(BaseRanker):
    def __init__(self, embedding_weight: float, bm25_weight: float):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        self.bm25: BM25Okapi

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        texts = [chunk.content for chunk in chunks]
        tokens = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokens)
        logger.debug(f"Built BM25 index with {len(chunks)} chunks")

    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        embedding_scores = (embeddings @ query_embedding.T).flatten()
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_scores_norm = bm25_scores / (np.max(bm25_scores) + 1e-8)

        combined_scores = (
            self.embedding_weight * embedding_scores
            + self.bm25_weight * bm25_scores_norm
        )

        ranked_indices = combined_scores.argsort()[::-1]

        return [
            (
                int(idx),
                float(combined_scores[idx]),
                float(embedding_scores[idx]),
                float(bm25_scores_norm[idx]),
            )
            for idx in ranked_indices
        ]
