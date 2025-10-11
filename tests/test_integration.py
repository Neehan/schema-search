import os
from pathlib import Path
import gc
from typing import cast

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine
import psutil

from schema_search import SchemaSearch
from schema_search.types import SearchType


@pytest.fixture(scope="module")
def database_url():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set in tests/.env file")

    return url


@pytest.fixture(scope="module")
def llm_config():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("LLM_API_KEY")
    base_url = "https://api.anthropic.com/v1/"

    if not api_key:
        pytest.skip("LLM_API_KEY not set in tests/.env file")

    return {"api_key": api_key, "base_url": base_url}


@pytest.fixture(scope="module")
def search_engine(database_url, llm_config):
    engine = create_engine(database_url)
    search = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )
    return search


def test_index_creation(search_engine):
    """Test that the index can be built successfully."""
    stats = search_engine.index(force=True)

    assert len(search_engine.schemas) > 0, "No tables found in database"
    assert len(search_engine.chunks) > 0, "No chunks generated"

    print(f"\nIndexing: {stats}")


def test_search_user_information(search_engine):
    """Test searching for user-related information in the schema."""
    search_engine.index(force=False)

    query = "which table has user email address?"
    response = search_engine.search(query)

    results = response["results"]

    for result in results:
        print(f"Result: {result['table']} (score: {result['score']:.3f})")
        # print(f"Related tables: {result['related_tables']}")
        # print("-" * 100)

    assert len(results) > 0, "No search results returned"

    top_result = results[0]
    assert "table" in top_result, "Result missing 'table' field"
    assert "score" in top_result, "Result missing 'score' field"
    assert "schema" in top_result, "Result missing 'schema' field"
    assert "matched_chunks" in top_result, "Result missing 'matched_chunks' field"
    assert "related_tables" in top_result, "Result missing 'related_tables' field"

    assert top_result["score"] > 0, "Top result has invalid score"

    print(f"\nTop result: {top_result['table']} (score: {top_result['score']:.3f})")
    print(f"Related tables: {top_result['related_tables']}")
    print(f"Search latency: {response['latency_sec']}s")


def _calculate_score(results, correct_table):
    """Calculate score based on position. Top=5, 2nd=4, 3rd=3, 4th=2, 5th=1, not found=0"""
    for position, result in enumerate(results[:5], 1):
        if result["table"] == correct_table:
            return 6 - position
    return 0


def _get_eval_data():
    """Return evaluation dataset."""
    return [
        {
            "question": "which table has user email address?",
            "correct_table": "user_metadata",
        },
        {
            "question": "which table has scrapped project content?",
            "correct_table": "project_content",
        },
        {
            "question": "where can I find complete list of twitter bot accounts?",
            "correct_table": "agent_metadata",
        },
        {
            "question": "which table user api keys??",
            "correct_table": "api_token",
        },
        {
            "question": "which table has user deposits?",
            "correct_table": "user_deposits",
        },
        {
            "question": "which table has information about infrastructure?",
            "correct_table": "node_metadata",
        },
        {
            "question": "which table has information about user balances?",
            "correct_table": "user_balances",
        },
        {
            "question": "which table maps news to topics?",
            "correct_table": "news_to_topic_map",
        },
        {
            "question": "which table has information about projects?",
            "correct_table": "project_metadata",
        },
        {
            "question": "which table user query metrics?",
            "correct_table": "query_metrics",
        },
    ]


def test_memory_bm25_isolated(database_url, llm_config):
    """Measure BM25 in complete isolation."""
    _run_memory_test_for_strategy(database_url, llm_config, "bm25")


def test_memory_fuzzy_isolated(database_url, llm_config):
    """Measure Fuzzy in complete isolation."""
    _run_memory_test_for_strategy(database_url, llm_config, "fuzzy")


def test_memory_semantic_isolated(database_url, llm_config):
    """Measure Semantic in complete isolation."""
    _run_memory_test_for_strategy(database_url, llm_config, "semantic")


def test_memory_hybrid_isolated(database_url, llm_config):
    """Measure Hybrid in complete isolation."""
    _run_memory_test_for_strategy(database_url, llm_config, "hybrid")


def _run_memory_test_for_strategy(database_url, llm_config, strategy):
    """Run memory test for a single strategy."""
    gc.collect()

    engine = create_engine(database_url)
    search_engine = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )

    search_engine.index(force=False)

    process = psutil.Process()
    after_index_mem = process.memory_info().rss / 1024 / 1024
    peak_memory = after_index_mem

    eval_data = _get_eval_data()
    memory_samples = []
    latency_samples = []
    total_score = 0

    print(f"\n{'='*50} {strategy.upper()} {'='*50}")
    print(f"After index: {after_index_mem:.2f} MB")
    print(f"Embedding cache created: {search_engine._embedding_cache is not None}")
    print(f"BM25 cache created: {search_engine._bm25_cache is not None}")

    for idx, eval_item in enumerate(eval_data, 1):
        question = eval_item["question"]
        correct_table = eval_item["correct_table"]

        before_mem = process.memory_info().rss / 1024 / 1024
        response = search_engine.search(
            question, search_type=cast(SearchType, strategy), hops=1
        )
        after_mem = process.memory_info().rss / 1024 / 1024

        peak_memory = max(peak_memory, after_mem)
        memory_samples.append(after_mem)
        latency_samples.append(response["latency_sec"])

        score = _calculate_score(response["results"], correct_table)
        total_score += score

        marker = "✓" if score > 0 else "✗"
        print(
            f"  Q{idx}: {marker} Score: {score} | "
            f"Latency: {response['latency_sec']:.3f}s | "
            f"Mem: {after_mem:.1f}MB ({after_mem - before_mem:+.1f})"
        )

    avg_memory = sum(memory_samples) / len(memory_samples)
    avg_latency = sum(latency_samples) / len(latency_samples)
    memory_increase = peak_memory - after_index_mem
    max_score = len(eval_data) * 5

    print(f"\n{'='*50} SUMMARY {'='*50}")
    print(f"Score: {total_score}/{max_score}")
    print(f"Avg Latency: {avg_latency:.3f}s")
    print(f"Peak Memory: {peak_memory:.2f} MB")
    print(f"Avg Memory: {avg_memory:.2f} MB")
    print(f"Memory Increase: +{memory_increase:.2f} MB")
    if search_engine._embedding_cache:
        print(
            f"Embeddings loaded: {search_engine._embedding_cache.embeddings is not None}"
        )
    if search_engine._bm25_cache:
        print(f"BM25 built: {search_engine._bm25_cache.bm25 is not None}")
    print("=" * 100)


def test_bm25_no_embeddings(database_url, llm_config):
    """Test that BM25 search does NOT load embedding models or cache."""
    engine = create_engine(database_url)
    search = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )

    search.index(force=False)

    assert search._embedding_cache is None, "Embedding cache should not be created yet"
    assert search._reranker is None, "Reranker should not be created yet"

    result = search.search("user email", search_type="bm25", limit=5)

    assert search._embedding_cache is None, "BM25 should not load embedding cache"
    assert len(result["results"]) > 0, "Should have results"

    print("\n✓ BM25 search verified: no embeddings loaded")


def test_fuzzy_no_embeddings(database_url, llm_config):
    """Test that fuzzy search does NOT load embedding models or cache."""
    engine = create_engine(database_url)
    search = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )

    search.index(force=False)

    assert search._embedding_cache is None, "Embedding cache should not be created yet"
    assert search._reranker is None, "Reranker should not be created yet"

    result = search.search("user email", search_type="fuzzy", limit=5)

    assert search._embedding_cache is None, "Fuzzy should not load embedding cache"
    assert len(result["results"]) > 0, "Should have results"

    print("\n✓ Fuzzy search verified: no embeddings loaded")


def test_semantic_loads_embeddings(database_url, llm_config):
    """Test that semantic search DOES load embedding models and cache."""
    engine = create_engine(database_url)
    search = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )

    search.index(force=False)

    assert search._embedding_cache is None, "Embedding cache should not be created yet"

    result = search.search("user email", search_type="semantic", limit=5)

    assert search._embedding_cache is not None, "Semantic should create embedding cache"
    assert search.embedding_cache.embeddings is not None, "Embeddings should be loaded"
    assert len(result["results"]) > 0, "Should have results"

    print("\n✓ Semantic search verified: embeddings loaded correctly")


def test_hybrid_loads_embeddings(database_url, llm_config):
    """Test that hybrid search DOES load embedding models and cache."""
    engine = create_engine(database_url)
    search = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )

    search.index(force=False)

    assert search._embedding_cache is None, "Embedding cache should not be created yet"

    result = search.search("user email", search_type="hybrid", limit=5)

    assert search._embedding_cache is not None, "Hybrid should create embedding cache"
    assert search.embedding_cache.embeddings is not None, "Embeddings should be loaded"
    assert len(result["results"]) > 0, "Should have results"

    print("\n✓ Hybrid search verified: embeddings loaded correctly")


def test_strategy_caching(database_url, llm_config):
    """Test that search strategies are cached and reused."""
    engine = create_engine(database_url)
    search = SchemaSearch(
        engine,
        llm_api_key=llm_config["api_key"],
        llm_base_url=llm_config["base_url"],
    )

    search.index(force=False)

    assert len(search._search_strategies) == 0, "No strategies cached initially"

    search.search("test query", search_type="bm25", limit=5)
    assert "bm25" in search._search_strategies, "BM25 strategy should be cached"
    assert len(search._search_strategies) == 1, "Only one strategy cached"

    bm25_strategy = search._search_strategies["bm25"]
    search.search("another query", search_type="bm25", limit=5)
    assert (
        search._search_strategies["bm25"] is bm25_strategy
    ), "Same strategy instance should be reused"

    search.search("test query", search_type="fuzzy", limit=5)
    assert "fuzzy" in search._search_strategies, "Fuzzy strategy should be cached"
    assert len(search._search_strategies) == 2, "Two strategies cached now"

    print("\n✓ Strategy caching verified: strategies are reused")
