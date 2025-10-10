from typing import Dict, List, Set, Callable

import numpy as np

from schema_search.search.base import BaseSearchStrategy
from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder
from schema_search.embedding_cache import EmbeddingCache
from schema_search.rankers import BaseRanker


class SemanticSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        embedding_cache: EmbeddingCache,
        initial_top_k: int,
        rerank_top_k: int,
        reranker_factory: Callable[[], BaseRanker],
    ):
        self.embedding_cache = embedding_cache
        self.initial_top_k = initial_top_k
        self.rerank_top_k = rerank_top_k
        self.reranker_factory = reranker_factory

    def search(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
        limit: int,
    ) -> List[SearchResultItem]:
        query_embedding = self.embedding_cache.encode_query(query)
        initial_tables = self._initial_ranking(query, query_embedding, chunks)
        expanded_tables = self._expand_graph(initial_tables, graph_builder, hops)
        return self._rerank(
            query,
            query_embedding,
            schemas,
            chunks,
            expanded_tables,
            graph_builder,
            hops,
            limit,
        )

    def _initial_ranking(
        self, query: str, query_embedding: np.ndarray, chunks: List[Chunk]
    ) -> Set[str]:
        embedding_scores = (
            self.embedding_cache.embeddings @ query_embedding.T
        ).flatten()
        top_indices = embedding_scores.argsort()[::-1][: self.initial_top_k]

        initial_tables: Set[str] = set()
        for idx in top_indices:
            initial_tables.add(chunks[idx].table_name)

        return initial_tables

    def _expand_graph(
        self, initial_tables: Set[str], graph_builder: GraphBuilder, hops: int
    ) -> Set[str]:
        expanded_tables = set(initial_tables)
        if hops == 0:
            return expanded_tables
        for table in initial_tables:
            neighbors = graph_builder.get_neighbors(table, hops)
            expanded_tables.update(neighbors)
        return expanded_tables

    def _rerank(
        self,
        query: str,
        query_embedding: np.ndarray,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        expanded_tables: Set[str],
        graph_builder: GraphBuilder,
        hops: int,
        limit: int,
    ) -> List[SearchResultItem]:
        table_chunks: Dict[str, List[int]] = {}
        for idx, chunk in enumerate(chunks):
            if chunk.table_name in expanded_tables:
                if chunk.table_name not in table_chunks:
                    table_chunks[chunk.table_name] = []
                table_chunks[chunk.table_name].append(idx)

        rerank_chunk_indices: List[int] = []
        for table_name in expanded_tables:
            rerank_chunk_indices.extend(table_chunks[table_name])

        rerank_embeddings = self.embedding_cache.embeddings[rerank_chunk_indices]
        rerank_chunks = [chunks[idx] for idx in rerank_chunk_indices]

        reranker = self.reranker_factory()
        reranker.build(rerank_chunks)
        reranked = reranker.rank(query, query_embedding, rerank_embeddings)

        final_table_chunk_map = reranker.get_top_tables_from_chunks(
            reranked, self.rerank_top_k
        )

        chunk_idx_to_score: Dict[int, float] = {}
        for chunk_idx, score, emb_score, aux_score in reranked:
            chunk_idx_to_score[chunk_idx] = score

        results: List[SearchResultItem] = []
        for table_name, chunk_indices in final_table_chunk_map.items():
            max_score = max(chunk_idx_to_score[idx] for idx in chunk_indices)
            matched_chunks = [rerank_chunks[idx].content for idx in chunk_indices]

            result = self._build_result_item(
                table_name=table_name,
                score=float(max_score),
                schema=schemas[table_name],
                matched_chunks=matched_chunks,
                graph_builder=graph_builder,
                hops=hops,
            )
            results.append(result)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
