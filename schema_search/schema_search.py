import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from sqlalchemy.engine import Engine

from schema_search.schema_extractor import SchemaExtractor
from schema_search.chunkers import Chunk, create_chunker
from schema_search.embedding_cache import EmbeddingCache
from schema_search.graph_builder import GraphBuilder
from schema_search.search import create_semantic_strategy, create_fuzzy_strategy
from schema_search.types import IndexResult, SearchResult, SearchType, TableSchema

logger = logging.getLogger(__name__)


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        if isinstance(result, dict):
            result["latency_sec"] = round(elapsed, 3)
        return result

    return wrapper


class SchemaSearch:
    def __init__(self, engine: Engine, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._setup_logging()

        cache_dir = Path(self.config["embedding"]["cache_dir"])
        cache_dir.mkdir(exist_ok=True)

        self.schemas: Dict[str, TableSchema] = {}
        self.chunks: List[Chunk] = []

        self.schema_extractor = SchemaExtractor(engine, self.config)
        self.chunker = create_chunker(self.config)
        self.embedding_cache = EmbeddingCache(
            cache_dir=cache_dir,
            model_name=self.config["embedding"]["model"],
            batch_size=self.config["embedding"]["batch_size"],
            normalize=self.config["embedding"]["normalize"],
            show_progress=self.config["embedding"]["show_progress"],
        )
        self.graph_builder = GraphBuilder(cache_dir)

        self.semantic_strategy = create_semantic_strategy(
            self.config, self.embedding_cache
        )
        self.fuzzy_strategy = create_fuzzy_strategy()

    def _setup_logging(self) -> None:
        level = getattr(logging, self.config["logging"]["level"])
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )
        logger.setLevel(level)

    def _load_config(self, config_path: Optional[str]) -> Dict:
        if config_path is None:
            config_path = str(Path(__file__).parent.parent / "config.yml")

        with open(config_path) as f:
            return yaml.safe_load(f)

    @time_it
    def index(self, force: bool = False) -> IndexResult:
        logger.info("Starting schema indexing" + (" (force)" if force else ""))

        self.schemas = self._load_or_extract_schemas(force)
        self.graph_builder.build(self.schemas, force)
        self.chunks = self._load_or_generate_chunks(self.schemas, force)
        self.embedding_cache.load_or_generate(
            self.chunks, force, self.config["chunking"]
        )

        logger.info(
            f"Indexing complete: {len(self.schemas)} tables, {len(self.chunks)} chunks"
        )
        return {
            "tables": len(self.schemas),
            "chunks": len(self.chunks),
            "latency_sec": 0.0,
        }

    def _load_or_extract_schemas(self, force: bool) -> Dict[str, TableSchema]:
        cache_dir = Path(self.config["embedding"]["cache_dir"])
        schema_cache = cache_dir / "metadata.json"

        if not force and schema_cache.exists():
            logger.debug(f"Loading schemas from cache: {schema_cache}")
            with open(schema_cache) as f:
                return json.load(f)

        logger.info("Extracting schema from database")
        schemas = self.schema_extractor.extract()

        with open(schema_cache, "w") as f:
            json.dump(schemas, f, indent=2)

        return schemas

    def _load_or_generate_chunks(
        self, schemas: Dict[str, TableSchema], force: bool
    ) -> List[Chunk]:
        cache_dir = Path(self.config["embedding"]["cache_dir"])
        chunks_cache = cache_dir / "chunk_metadata.json"

        if not force and chunks_cache.exists():
            logger.info(f"Loading chunks from cache: {chunks_cache}")
            with open(chunks_cache) as f:
                chunk_data = json.load(f)
                return [
                    Chunk(
                        table_name=c["table_name"],
                        content=c["content"],
                        chunk_id=c["chunk_id"],
                        token_count=c["token_count"],
                    )
                    for c in chunk_data
                ]

        logger.info("Generating chunks from schemas")
        chunks = self.chunker.chunk_schemas(schemas)

        with open(chunks_cache, "w") as f:
            chunk_data = [
                {
                    "table_name": c.table_name,
                    "content": c.content,
                    "chunk_id": c.chunk_id,
                    "token_count": c.token_count,
                }
                for c in chunks
            ]
            json.dump(chunk_data, f, indent=2)

        return chunks

    @time_it
    def search(
        self,
        query: str,
        hops: int = 1,
        limit: int = 5,
        search_type: SearchType = "semantic",
    ) -> SearchResult:
        logger.debug(f"Searching: {query} (hops={hops}, search_type={search_type})")

        if search_type == "semantic":
            strategy = self.semantic_strategy
        elif search_type == "fuzzy":
            strategy = self.fuzzy_strategy
        else:
            raise ValueError(f"Invalid search_type: {search_type}")

        results = strategy.search(
            query, self.schemas, self.chunks, self.graph_builder, hops, limit
        )

        logger.debug(f"Found {len(results)} results")

        return {"results": results, "latency_sec": 0.0}
