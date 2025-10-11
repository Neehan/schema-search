from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder
from schema_search.embedding_cache import BaseEmbeddingCache
from schema_search.rankers.base import BaseRanker

if TYPE_CHECKING:
    from schema_search.embedding_cache.bm25 import BM25Cache


class HybridSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        embedding_cache: BaseEmbeddingCache,
        bm25_cache: "BM25Cache",
        initial_top_k: int,
        rerank_top_k: int,
        reranker: Optional[BaseRanker],
        semantic_weight: float,
    ):
        super().__init__(reranker, initial_top_k, rerank_top_k)
        assert 0 <= semantic_weight <= 1, "semantic_weight must be between 0 and 1"
        self.embedding_cache = embedding_cache
        self.bm25_cache = bm25_cache
        self.semantic_weight = semantic_weight
        self.bm25_weight = 1 - semantic_weight

    def _initial_ranking(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        query_embedding = self.embedding_cache.encode_query(query)
        semantic_scores = self.embedding_cache.compute_similarities(query_embedding)

        bm25_scores = self.bm25_cache.get_scores(query)

        semantic_min = semantic_scores.min()
        semantic_max = semantic_scores.max()
        semantic_range = semantic_max - semantic_min
        if semantic_range > 0:
            semantic_scores_norm = (semantic_scores - semantic_min) / semantic_range
        else:
            semantic_scores_norm = np.zeros_like(semantic_scores)

        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        bm25_range = bm25_max - bm25_min
        if bm25_range > 0:
            bm25_scores_norm = (bm25_scores - bm25_min) / bm25_range
        else:
            bm25_scores_norm = np.zeros_like(bm25_scores)

        hybrid_scores = (
            self.semantic_weight * semantic_scores_norm
            + self.bm25_weight * bm25_scores_norm
        )

        top_indices = hybrid_scores.argsort()[::-1][: self.initial_top_k]

        results: List[SearchResultItem] = []
        for idx in top_indices:
            chunk = chunks[idx]
            result = self._build_result_item(
                table_name=chunk.table_name,
                score=float(hybrid_scores[idx]),
                schema=schemas[chunk.table_name],
                matched_chunks=[chunk.content],
                graph_builder=graph_builder,
                hops=hops,
            )
            results.append(result)

        return results
