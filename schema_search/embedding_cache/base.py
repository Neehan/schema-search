from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np

from schema_search.chunkers import Chunk


class BaseEmbeddingCache(ABC):
    def __init__(
        self,
        cache_dir: Path,
        model_name: str,
        metric: str,
        batch_size: int,
        show_progress: bool,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        self.model_name = model_name
        self.model = None
        self.metric = metric
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.embeddings = None

    @abstractmethod
    def load_or_generate(
        self, chunks: List[Chunk], force: bool, chunking_config: Dict
    ) -> None:
        pass

    @abstractmethod
    def encode_query(self, query: str) -> np.ndarray:
        pass

    @abstractmethod
    def compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        pass
