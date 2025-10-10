import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from schema_search.chunkers import Chunk

logger = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        batch_size: int,
        normalize: bool,
        show_progress: bool,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        self.model: SentenceTransformer
        self.embeddings: np.ndarray

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

        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
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
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        self.model = SentenceTransformer(self.model_name)

    def encode_query(self, query: str) -> np.ndarray:
        self._load_model()

        query_emb = self.model.encode(
            [query],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
        )

        return query_emb
