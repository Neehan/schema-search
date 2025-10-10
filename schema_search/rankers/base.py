from typing import Dict, List, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np

from schema_search.chunkers import Chunk


class BaseRanker(ABC):
    def __init__(self):
        self.chunks: List[Chunk]

    @abstractmethod
    def build(self, chunks: List[Chunk]) -> None:
        pass

    @abstractmethod
    def rank(
        self, query: str, query_embedding: np.ndarray, embeddings: np.ndarray
    ) -> List[Tuple[int, float, float, float]]:
        """
        Returns: List of (chunk_idx, combined_score, embedding_score, auxiliary_score)
        """
        pass

    def get_top_tables_from_chunks(
        self, ranked_chunks: List[Tuple[int, float, float, float]], top_k: int
    ) -> Dict[str, List[int]]:
        table_to_chunk_indices: Dict[str, List[int]] = defaultdict(list)
        chunk_idx_to_score: Dict[int, float] = {}

        for chunk_idx, score, emb_score, aux_score in ranked_chunks:
            chunk = self.chunks[chunk_idx]
            table_to_chunk_indices[chunk.table_name].append(chunk_idx)
            chunk_idx_to_score[chunk_idx] = score

        table_scores: Dict[str, float] = {}
        for table_name, chunk_indices in table_to_chunk_indices.items():
            max_score = max(chunk_idx_to_score[idx] for idx in chunk_indices)
            table_scores[table_name] = max_score

        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        result: Dict[str, List[int]] = {}
        for table_name, score in top_tables:
            result[table_name] = table_to_chunk_indices[table_name]

        return result
