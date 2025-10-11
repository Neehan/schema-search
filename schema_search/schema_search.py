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
from schema_search.embedding_cache import create_embedding_cache
from schema_search.embedding_cache.bm25 import BM25Cache
from schema_search.graph_builder import GraphBuilder
from schema_search.search import create_search_strategy
from schema_search.types import IndexResult, SearchResult, SearchType, TableSchema
from schema_search.rankers import create_ranker


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
    def __init__(
        self,
        engine: Engine,
        config_path: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        self.config = self._load_config(config_path)
        self._setup_logging()

        cache_dir = Path(self.config["embedding"]["cache_dir"])
        cache_dir.mkdir(exist_ok=True)

        self.schemas: Dict[str, TableSchema] = {}
        self.chunks: List[Chunk] = []
        self.cache_dir = cache_dir

        chunking_strategy = self.config["chunking"]["strategy"]
        if chunking_strategy == "llm" and not llm_api_key:
            raise ValueError(
                "LLM chunking strategy requires llm_api_key parameter. "
                "Pass it to SchemaSearch constructor."
            )

        self.schema_extractor = SchemaExtractor(engine, self.config)
        self.chunker = create_chunker(self.config, llm_api_key, llm_base_url)
        self._embedding_cache = None
        self._bm25_cache = None
        self.graph_builder = GraphBuilder(cache_dir)
        self._reranker = None
        self._reranker_config = self.config["reranker"]["model"]
        self._search_strategies = {}

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
        self._index_force = force

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

    def _get_embedding_cache(self):
        if self._embedding_cache is None:
            self._embedding_cache = create_embedding_cache(self.config, self.cache_dir)
        return self._embedding_cache

    def _get_reranker(self):
        if self._reranker is None and self._reranker_config:
            self._reranker = create_ranker(self.config)
        return self._reranker

    @property
    def embedding_cache(self):
        return self._get_embedding_cache()

    @property
    def reranker(self):
        return self._get_reranker()

    def _get_bm25_cache(self):
        if self._bm25_cache is None:
            self._bm25_cache = BM25Cache()
        return self._bm25_cache

    def _ensure_embeddings_loaded(self):
        cache = self._get_embedding_cache()
        if cache.embeddings is None:
            cache.load_or_generate(
                self.chunks, self._index_force, self.config["chunking"]
            )

    def _ensure_bm25_built(self):
        cache = self._get_bm25_cache()
        if cache.bm25 is None:
            logger.info("Building BM25 index")
            cache.build(self.chunks)

    def _get_search_strategy(self, search_type: str):
        if search_type not in self._search_strategies:
            self._search_strategies[search_type] = create_search_strategy(
                self.config,
                self._get_embedding_cache,
                self._get_bm25_cache,
                self._get_reranker,
                search_type,
            )
        return self._search_strategies[search_type]

    @time_it
    def search(
        self,
        query: str,
        hops: Optional[int] = None,
        limit: int = 5,
        search_type: Optional[SearchType] = None,
    ) -> SearchResult:
        if hops is None:
            hops = int(self.config["search"]["hops"])
        logger.debug(f"Searching: {query} (hops={hops}, search_type={search_type})")

        search_type = search_type or self.config["search"]["strategy"]

        if search_type in ["semantic", "hybrid"]:
            self._ensure_embeddings_loaded()

        if search_type in ["bm25", "hybrid"]:
            self._ensure_bm25_built()

        strategy = self._get_search_strategy(search_type)

        results = strategy.search(
            query, self.schemas, self.chunks, self.graph_builder, hops, limit
        )

        logger.debug(f"Found {len(results)} results")

        return {"results": results, "latency_sec": 0.0}
