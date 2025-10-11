from typing import Dict, List, Optional, TYPE_CHECKING

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder
from schema_search.rankers.base import BaseRanker

if TYPE_CHECKING:
    from schema_search.embedding_cache.bm25 import BM25Cache


class BM25SearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        bm25_cache: "BM25Cache",
        initial_top_k: int,
        rerank_top_k: int,
        reranker: Optional[BaseRanker],
    ):
        super().__init__(reranker, initial_top_k, rerank_top_k)
        self.bm25_cache = bm25_cache

    def _initial_ranking(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> List[SearchResultItem]:
        scores = self.bm25_cache.get_scores(query)
        top_indices = scores.argsort()[::-1][: self.initial_top_k]

        results: List[SearchResultItem] = []
        for idx in top_indices:
            chunk = chunks[idx]
            result = self._build_result_item(
                table_name=chunk.table_name,
                score=float(scores[idx]),
                schema=schemas[chunk.table_name],
                matched_chunks=[chunk.content],
                graph_builder=graph_builder,
                hops=hops,
            )
            results.append(result)

        return results
