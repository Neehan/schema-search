from typing import Callable, Dict, Optional

from schema_search.search.semantic import SemanticSearchStrategy
from schema_search.search.fuzzy import FuzzySearchStrategy
from schema_search.search.bm25 import BM25SearchStrategy
from schema_search.search.hybrid import HybridSearchStrategy
from schema_search.search.base import BaseSearchStrategy
from schema_search.embedding_cache import BaseEmbeddingCache
from schema_search.rankers.base import BaseRanker


def create_search_strategy(
    config: Dict,
    get_embedding_cache: Callable[[], BaseEmbeddingCache],
    get_bm25_cache: Callable,
    get_reranker: Callable[[], Optional[BaseRanker]],
    strategy_type: Optional[str],
) -> BaseSearchStrategy:
    search_config = config["search"]
    strategy_type = strategy_type or search_config["strategy"]

    initial_top_k = search_config["initial_top_k"]
    rerank_top_k = search_config["rerank_top_k"]

    reranker = get_reranker()

    if strategy_type == "semantic":
        return SemanticSearchStrategy(
            embedding_cache=get_embedding_cache(),
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            reranker=reranker,
        )

    if strategy_type == "bm25":
        return BM25SearchStrategy(
            bm25_cache=get_bm25_cache(),
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            reranker=reranker,
        )

    if strategy_type == "fuzzy":
        return FuzzySearchStrategy(
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            reranker=reranker,
        )

    if strategy_type == "hybrid":
        semantic_weight = search_config["semantic_weight"]
        return HybridSearchStrategy(
            embedding_cache=get_embedding_cache(),
            bm25_cache=get_bm25_cache(),
            initial_top_k=initial_top_k,
            rerank_top_k=rerank_top_k,
            reranker=reranker,
            semantic_weight=semantic_weight,
        )

    raise ValueError(f"Unknown search strategy: {strategy_type}")
