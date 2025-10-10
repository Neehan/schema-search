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
def search_engine(database_url):
    engine = create_engine(database_url)
    search = SchemaSearch(engine)
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


def _print_results(label, results, correct_table, score):
    """Print search results with score."""
    print(f"\n{label} - Score: {score}/5")
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
    """Compare search results: semantic without graph, semantic with graph, and fuzzy."""
    search_engine.index(force=False)

    hops_with_graph = 1
    eval_data = _get_eval_data()

    print("\n" + "=" * 100)
    print("EVALUATION: Search Method Comparison")
    print("=" * 100)
    print("Comparing: Semantic (hops=0) vs Semantic (hops=1) vs Fuzzy")
    print(
        "Scoring: Rank 1=5pts, Rank 2=4pts, Rank 3=3pts, Rank 4=2pts, Rank 5=1pt, Not found=0pts"
    )
    print("=" * 100)

    total_scores = {"semantic_no_graph": 0, "semantic_with_graph": 0, "fuzzy": 0}

    for idx, eval_item in enumerate(eval_data, 1):
        question = eval_item["question"]
        correct_table = eval_item["correct_table"]

        response_no_graph = search_engine.search(
            question, hops=0, search_type="semantic"
        )
        response_with_graph = search_engine.search(
            question, hops=hops_with_graph, search_type="semantic"
        )
        response_fuzzy = search_engine.search(question, search_type="fuzzy")

        results_no_graph = response_no_graph["results"]
        results_with_graph = response_with_graph["results"]
        results_fuzzy = response_fuzzy["results"]

        score_no_graph = _calculate_score(results_no_graph, correct_table)
        score_with_graph = _calculate_score(results_with_graph, correct_table)
        score_fuzzy = _calculate_score(results_fuzzy, correct_table)

        total_scores["semantic_no_graph"] += score_no_graph
        total_scores["semantic_with_graph"] += score_with_graph
        total_scores["fuzzy"] += score_fuzzy

        print(f"\n--- Question {idx} ---")
        print(f"Q: {question}")
        print(f"Correct Answer: {correct_table}")

        _print_results(
            "Semantic (hops=0)", results_no_graph, correct_table, score_no_graph
        )
        _print_results(
            f"Semantic (hops={hops_with_graph})",
            results_with_graph,
            correct_table,
            score_with_graph,
        )
        _print_results("Fuzzy", results_fuzzy, correct_table, score_fuzzy)

    print("\n" + "=" * 100)
    print("FINAL SCORES")
    print("=" * 100)
    max_possible_score = len(eval_data) * 5
    print(
        f"Semantic (hops=0):      {total_scores['semantic_no_graph']}/{max_possible_score}"
    )
    print(
        f"Semantic (hops=1):      {total_scores['semantic_with_graph']}/{max_possible_score}"
    )
    print(f"Fuzzy:                  {total_scores['fuzzy']}/{max_possible_score}")
    print("=" * 100)

    assert len(eval_data) > 0, "No evaluation data provided"
    for score in total_scores.values():
        assert score >= 0, "Invalid score"
