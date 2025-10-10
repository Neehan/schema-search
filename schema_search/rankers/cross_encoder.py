from typing import List, Tuple
import logging

import numpy as np
from sentence_transformers import CrossEncoder

from schema_search.chunkers import Chunk
from schema_search.rankers.base import BaseRanker

logger = logging.getLogger(__name__)


class CrossEncoderRanker(BaseRanker):
    def __init__(self, model_name: str, initial_top_k: int):
        super().__init__()
        self.model_name = model_name
        self.initial_top_k = initial_top_k
        self.model = None

    def _load_model(self) -> CrossEncoder:
        if self.model is None:
            import logging as log
            log.getLogger("sentence_transformers").setLevel(log.WARNING)
            self.model = CrossEncoder(self.model_name)
            logger.info(f"Loaded CrossEncoder: {self.model_name}")
        return self.model

    def build(self, chunks: List[Chunk]) -> None:
        self.chunks = chunks
        logger.debug(f"Initialized CrossEncoder reranker with {len(chunks)} chunks")

    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        model = self._load_model()

        embedding_scores = (embeddings @ query_embedding.T).flatten()
        top_indices = embedding_scores.argsort()[::-1][: self.initial_top_k]

        pairs = [(query, self.chunks[idx].content) for idx in top_indices]
        rerank_scores = model.predict(pairs, show_progress_bar=False)

        ranked_indices = top_indices[rerank_scores.argsort()[::-1]]

        return [
            (
                int(idx),
                float(rerank_scores[i]),
                float(embedding_scores[idx]),
                float(rerank_scores[i]),
            )
            for i, idx in enumerate(ranked_indices)
        ]
