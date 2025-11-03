import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np

from schema_search.chunkers import Chunk
from schema_search.embedding_cache.base import BaseEmbeddingCache
from schema_search.metrics import get_metric
from schema_search.utils.lazy_import import lazy_import_check

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class InMemoryEmbeddingCache(BaseEmbeddingCache):
    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        metric: str,
        batch_size: int,
        show_progress: bool,
    ):
        super().__init__(cache_dir, model_name, metric, batch_size, show_progress)
        self.model: Optional["SentenceTransformer"] = None

    def load_or_generate(
        self, chunks: List[Chunk], force: bool, chunking_config: Dict
    ) -> None:
        cache_file = self.cache_dir / "embeddings.npz"
        config_file = self.cache_dir / "cache_config.json"

        if not force and self._is_cache_valid(cache_file, config_file, chunking_config):
            self._load_from_cache(cache_file)
        else:
            self._generate_and_cache(chunks, cache_file, config_file, chunking_config)

    def _load_from_cache(self, cache_file: Path) -> None:
        logger.info("Loading embeddings from cache")
        self.embeddings = np.load(cache_file)["embeddings"]

    def _is_cache_valid(
        self, cache_file: Path, config_file: Path, chunking_config: Dict
    ) -> bool:
        if not (cache_file.exists() and config_file.exists()):
            return False

        with open(config_file) as f:
            cached_config = json.load(f)

        current_config = {
            "strategy": chunking_config["strategy"],
            "max_tokens": chunking_config["max_tokens"],
            "embedding_model": self.model_name,
        }

        if cached_config != current_config:
            logger.info("Cache invalidated: chunking config changed")
            return False

        return True

    def _generate_and_cache(
        self,
        chunks: List[Chunk],
        cache_file: Path,
        config_file: Path,
        chunking_config: Dict,
    ) -> None:
        self._load_model()

        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        texts = [chunk.content for chunk in chunks]

        assert self.model is not None
        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress,
        )

        np.savez_compressed(cache_file, embeddings=self.embeddings)

        cache_config = {
            "strategy": chunking_config["strategy"],
            "max_tokens": chunking_config["max_tokens"],
            "embedding_model": self.model_name,
        }
        with open(config_file, "w") as f:
            json.dump(cache_config, f, indent=2)

    def _load_model(self) -> None:
        if self.model is None:
            sentence_transformers = lazy_import_check(
                "sentence_transformers",
                "semantic",
                "semantic/hybrid search or reranking",
            )
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            self.model = sentence_transformers.SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()

        assert self.model is not None
        query_emb = self.model.encode(
            [query],
            batch_size=self.batch_size,
            normalize_embeddings=True,
        )

        return query_emb

    def compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        metric_fn = get_metric(self.metric)
        return metric_fn(self.embeddings, query_embedding).flatten()
