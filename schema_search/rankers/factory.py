from typing import Dict, Callable

from schema_search.rankers.base import BaseRanker
from schema_search.rankers.bm25 import BM25Ranker
from schema_search.rankers.cross_encoder import CrossEncoderRanker


def create_ranker_factory(config: Dict) -> Callable[[], BaseRanker]:
    reranker_config = config["reranker"]
    reranker_strategy = reranker_config["strategy"]

    if reranker_strategy == "cross_encoder":
        return lambda: CrossEncoderRanker(
            model_name=reranker_config["model"],
            initial_top_k=config["search"]["initial_top_k"],
        )
    elif reranker_strategy == "bm25_hybrid":
        return lambda: BM25Ranker(
            embedding_weight=reranker_config["embedding_weight"],
            bm25_weight=reranker_config["bm25_weight"],
        )
    else:
        raise ValueError(f"Unknown reranker strategy: {reranker_strategy}")
