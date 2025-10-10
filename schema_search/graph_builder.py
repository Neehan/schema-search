import logging
import pickle
from pathlib import Path
from typing import Dict, Set

import networkx as nx

from schema_search.types import TableSchema

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.graph: nx.DiGraph

    def build(self, schemas: Dict[str, TableSchema], force: bool) -> None:
        cache_file = self.cache_dir / "graph.pkl"

        if not force and cache_file.exists():
            self._load_from_cache(cache_file)
        else:
            self._build_and_cache(schemas, cache_file)

    def _load_from_cache(self, cache_file: Path) -> None:
        logger.debug(f"Loading graph from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            self.graph = pickle.load(f)

    def _build_and_cache(
        self, schemas: Dict[str, TableSchema], cache_file: Path
    ) -> None:
        logger.info("Building foreign key relationship graph")
        self.graph = nx.DiGraph()

        for table_name, schema in schemas.items():
            self.graph.add_node(table_name, **schema)

        for table_name, schema in schemas.items():
            if schema["foreign_keys"]:
                for fk in schema["foreign_keys"]:
                    referred_table = fk["referred_table"]
                    if referred_table in self.graph:
                        self.graph.add_edge(table_name, referred_table, **fk)

        with open(cache_file, "wb") as f:
            pickle.dump(self.graph, f)

    def get_neighbors(self, table_name: str, hops: int) -> Set[str]:
        if table_name not in self.graph:
            return set()

        neighbors: Set[str] = set()

        forward = nx.single_source_shortest_path_length(
            self.graph, table_name, cutoff=hops
        )
        neighbors.update(forward.keys())

        backward = nx.single_source_shortest_path_length(
            self.graph.reverse(), table_name, cutoff=hops
        )
        neighbors.update(backward.keys())

        neighbors.discard(table_name)

        return neighbors
