from typing import Dict

from schema_search.search.semantic import SemanticSearchStrategy
from schema_search.search.fuzzy import FuzzySearchStrategy
from schema_search.search.base import BaseSearchStrategy
from schema_search.embedding_cache import EmbeddingCache
from schema_search.rankers import create_ranker_factory


def create_semantic_strategy(
    config: Dict, embedding_cache: EmbeddingCache
) -> SemanticSearchStrategy:
    reranker_factory = create_ranker_factory(config)
    return SemanticSearchStrategy(
        embedding_cache=embedding_cache,
        initial_top_k=config["search"]["initial_top_k"],
        rerank_top_k=config["search"]["rerank_top_k"],
        reranker_factory=reranker_factory,
    )


def create_fuzzy_strategy() -> FuzzySearchStrategy:
    return FuzzySearchStrategy()
