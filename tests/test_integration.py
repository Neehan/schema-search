import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine

from schema_search import SchemaSearch


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
    assert (
        search_engine.embedding_cache.embeddings is not None
    ), "Embeddings not generated"

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


def _print_results(label, results, correct_table, score, latency):
    """Print search results with score."""
    print(f"\n{label} - Score: {score}/5 - Latency: {latency:.3f}s")
    for i, result in enumerate(results[:5], 1):
        marker = " ✓" if result["table"] == correct_table else ""
        print(f"  {i}. {result['table']} (score: {result['score']:.3f}){marker}")


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


def test_search_comparison_with_without_graph(search_engine):
    """Compare search results: semantic, BM25, fuzzy, and hybrid."""
    search_engine.index(force=True)

    eval_data = _get_eval_data()
    strategies = ["hybrid", "semantic", "bm25", "fuzzy"]

    print("\n" + "=" * 100)
    print("EVALUATION: Search Method Comparison")
    print("=" * 100)
    print("Comparing: Semantic vs BM25 vs Fuzzy vs Hybrid")
    print(
        "Scoring: Rank 1=5pts, Rank 2=4pts, Rank 3=3pts, Rank 4=2pts, Rank 5=1pt, Not found=0pts"
    )
    print("=" * 100)

    total_scores = {strategy: 0 for strategy in strategies}
    total_latencies = {strategy: 0.0 for strategy in strategies}

    for idx, eval_item in enumerate(eval_data, 1):
        question = eval_item["question"]
        correct_table = eval_item["correct_table"]

        print(f"\n--- Question {idx} ---")
        print(f"Q: {question}")
        print(f"Correct Answer: {correct_table}")

        for strategy in strategies:
            response = search_engine.search(question, search_type=strategy, hops=1)
            results = response["results"]
            score = _calculate_score(results, correct_table)

            total_scores[strategy] += score
            total_latencies[strategy] += response["latency_sec"]

            _print_results(
                strategy.capitalize(),
                results,
                correct_table,
                score,
                response["latency_sec"],
            )

    print("\n" + "=" * 100)
    print("FINAL SCORES")
    print("=" * 100)
    max_possible_score = len(eval_data) * 5
    num_questions = len(eval_data)

    for strategy in strategies:
        print(
            f"{strategy.capitalize():10s} {total_scores[strategy]}/{max_possible_score} "
            f"(avg latency: {total_latencies[strategy]/num_questions:.3f}s)"
        )
    print("=" * 100)

    assert len(eval_data) > 0, "No evaluation data provided"
    for score in total_scores.values():
        assert score >= 0, "Invalid score"
