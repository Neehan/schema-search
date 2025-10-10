from typing import Dict, List
from abc import ABC, abstractmethod

from schema_search.types import TableSchema, SearchResultItem
from schema_search.chunkers import Chunk
from schema_search.graph_builder import GraphBuilder


class BaseSearchStrategy(ABC):
    @abstractmethod
    def search(
        self,
        query: str,
        schemas: Dict[str, TableSchema],
        chunks: List[Chunk],
        graph_builder: GraphBuilder,
        hops: int,
        limit: int,
    ) -> List[SearchResultItem]:
        pass

    def _build_result_item(
        self,
        table_name: str,
        score: float,
        schema: TableSchema,
        matched_chunks: List[str],
        graph_builder: GraphBuilder,
        hops: int,
    ) -> SearchResultItem:
        return {
            "table": table_name,
            "score": score,
            "schema": schema,
            "matched_chunks": matched_chunks,
            "related_tables": list(graph_builder.get_neighbors(table_name, hops)),
        }
