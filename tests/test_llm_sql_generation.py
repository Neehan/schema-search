import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import anthropic
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
    base_url = os.getenv("LLM_BASE_URL", "https://api.anthropic.com")

    if not api_key:
        pytest.skip("LLM_API_KEY not set in tests/.env file")

    return {"api_key": api_key, "base_url": base_url}


@pytest.fixture(scope="module")
def search_engine(database_url):
    engine = create_engine(database_url)
    search = SchemaSearch(engine)
    search.index(force=False)
    return search


def test_table_identification_with_schema_search(search_engine, llm_config):
    """
    Compare table identification quality when LLM has:
    1. Full schema context (all tables and indices)
    2. Limited context from schema search with graph hops

    For each natural language question, we:
    - Ask LLM which tables are needed with full schema context (baseline)
    - Ask LLM which tables are needed with schema search context (our approach)
    - Compare both against the objective list of required tables
    """

    eval_data = [
        {
            "question": "how many unique users do we have?",
            "required_tables": ["user_metadata"],
            "searches": ["user table"],
            "hops": 1,
        },
        {
            "question": "what is the email of the user who deposited the most last month",
            "required_tables": ["user_metadata", "user_deposits"],
            "searches": ["user email deposit"],
            "hops": 1,
        },
        {
            "question": "what is the twitter handle of the agent that posted the most?",
            "required_tables": ["agent_metadata", "agent_content"],
            "searches": ["agent metadata content"],
            "hops": 1,
        },
        {
            "question": "which topic was covered the most in news articles last month?",
            "required_tables": ["news_to_topic_map", "topic_metadata"],
            "searches": ["topic metadata news map"],
            "hops": 1,
        },
        {
            "question": "which coin's price increased the most last month?",
            "required_tables": ["historical_market_data"],
            "searches": ["historical market data"],
            "hops": 1,
        },
        {
            "question": "find the 5 most recent news about the coin that increased the most last month?",
            "required_tables": [
                "historical_market_data",
                "news_to_topic_map",
                "topic_metadata",
                "news_summary",
            ],
            "searches": ["historical market data news topic"],
            "hops": 1,
        },
        {
            "question": "which model did the top user of last month use?",
            "required_tables": ["user_metadata", "model_metadata", "query_metrics"],
            "searches": ["user metadata model query metrics"],
            "hops": 1,
        },
        {
            "question": "which agent gained the most followers last month?",
            "required_tables": ["agent_metadata", "twitter_follow_activity"],
            "searches": ["agent metadata twitter follow activity"],
            "hops": 1,
        },
        {
            "question": "which agent posted the most content last month?",
            "required_tables": ["agent_metadata", "agent_content"],
            "searches": ["agent metadata agent content"],
            "hops": 1,
        },
        {
            "question": "which api key was most used during last month?",
            "required_tables": ["api_token", "query_metrics", "user_metadata"],
            "searches": ["api token query metrics user metadata"],
            "hops": 1,
        },
    ]

    def get_baseline_context(search_engine):
        """Get minimal context: just table names and indices."""
        context_parts = []

        for table_name, table_schema in search_engine.schemas.items():
            context_parts.append(f"Table: {table_name}")

            indices = table_schema.get("indices")
            if indices:
                idx_list = ", ".join([idx["name"] for idx in indices])
                context_parts.append(f"Indices: {idx_list}")

        return "\n\n".join(context_parts)

    def get_search_results_context(search_engine, searches, hops):
        """Get detailed schema from search results to add to baseline."""
        context_parts = []
        seen_tables = set()

        for search_query in searches:
            response = search_engine.search(
                search_query, hops=hops, limit=5, search_type="semantic"
            )
            for result in response["results"]:
                table_name = result["table"]
                if table_name in seen_tables:
                    continue
                seen_tables.add(table_name)

                columns = result["schema"].get("columns")
                if columns:
                    col_list = ", ".join(
                        [f"{col['name']} ({col['type']})" for col in columns]
                    )
                    context_parts.append(f"Table: {table_name}\nColumns: {col_list}")
        print("Search results tables: ", list(seen_tables))

        return "\n\n".join(context_parts)

    def call_llm_for_tables(question, schema_context, llm_config):
        """Call LLM to identify which tables are needed."""
        client = anthropic.Anthropic(api_key=llm_config["api_key"])

        prompt = f"""Given the following database schema:

{schema_context}

Which tables are necessary to answer this question: {question}

Return ONLY a comma-separated list of table names, nothing else. No explanations or additional text.
Example format: table1, table2, table3"""

        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=512,
            system="You are a database expert. Identify only the tables needed to answer the question.",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        tables_str = response.content[0].text.strip()  # type: ignore
        tables = [t.strip() for t in tables_str.split(",") if t.strip()]
        # Remove schema prefix if present
        tables = [t.split(".")[-1] for t in tables]
        return tables

    def compare_tables(identified_tables, required_tables):
        """Compare identified tables with required tables."""
        identified_set = set(t.lower() for t in identified_tables)
        required_set = set(t.lower() for t in required_tables)

        correct = identified_set & required_set
        missing = required_set - identified_set
        extra = identified_set - required_set

        is_perfect = len(missing) == 0 and len(extra) == 0

        return {
            "is_perfect": is_perfect,
            "correct": correct,
            "missing": missing,
            "extra": extra,
            "precision": len(correct) / len(identified_set) if identified_set else 0,
            "recall": len(correct) / len(required_set) if required_set else 0,
        }

    if len(eval_data) == 0:
        pytest.skip("No evaluation data provided")

    print("\n" + "=" * 100)
    print("EVALUATION: Table Identification - Baseline vs Baseline + Search Results")
    print("=" * 100)

    baseline_context = get_baseline_context(search_engine)

    baseline_perfect = 0
    baseline_total_precision = 0
    baseline_total_recall = 0

    search_perfect = 0
    search_total_precision = 0
    search_total_recall = 0

    for idx, eval_item in enumerate(eval_data, 1):
        question = eval_item["question"]
        required_tables = eval_item.get("required_tables", [])
        searches = eval_item.get("searches", [question])
        hops = eval_item.get("hops", 1)

        print(f"\n{'='*100}")
        print(f"Question {idx}: {question}")
        print(f"Required tables: {required_tables}")
        print(f"{'='*100}")

        # Get search results and combine with baseline
        search_results_context = get_search_results_context(
            search_engine, searches, hops
        )
        enhanced_context = baseline_context + "\n\n" + search_results_context

        print(f"\n[Baseline only] Context: {len(baseline_context)} chars")
        print(f"[Baseline + Search] Context: {len(enhanced_context)} chars")
        print(f"Additional context from search: {len(search_results_context)} chars")

        # Identify tables with baseline only
        print("\n--- Identifying tables with BASELINE ONLY ---")
        tables_baseline = call_llm_for_tables(question, baseline_context, llm_config)
        print(f"Identified tables: {tables_baseline}")

        comparison_baseline = compare_tables(tables_baseline, required_tables)
        print(
            f"Precision: {comparison_baseline['precision']:.2f}, Recall: {comparison_baseline['recall']:.2f}"
        )
        if comparison_baseline["missing"]:
            print(f"Missing: {comparison_baseline['missing']}")
        if comparison_baseline["extra"]:
            print(f"Extra: {comparison_baseline['extra']}")

        # Identify tables with baseline + search results
        print("\n--- Identifying tables with BASELINE + SEARCH ---")
        tables_search = call_llm_for_tables(question, enhanced_context, llm_config)
        print(f"Identified tables: {tables_search}")

        comparison_search = compare_tables(tables_search, required_tables)
        print(
            f"Precision: {comparison_search['precision']:.2f}, Recall: {comparison_search['recall']:.2f}"
        )
        if comparison_search["missing"]:
            print(f"Missing: {comparison_search['missing']}")
        if comparison_search["extra"]:
            print(f"Extra: {comparison_search['extra']}")

        # Track metrics
        if comparison_baseline["is_perfect"]:
            baseline_perfect += 1
            print("\n✓ Baseline: PERFECT")
        else:
            print("\n✗ Baseline: Not perfect")

        if comparison_search["is_perfect"]:
            search_perfect += 1
            print("✓ Schema Search: PERFECT")
        else:
            print("✗ Schema Search: Not perfect")

        baseline_total_precision += comparison_baseline["precision"]
        baseline_total_recall += comparison_baseline["recall"]
        search_total_precision += comparison_search["precision"]
        search_total_recall += comparison_search["recall"]

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)
    total_questions = len(eval_data)
    print(f"Total questions: {total_questions}")
    print(f"\nBaseline Only:")
    print(f"  Perfect matches: {baseline_perfect}/{total_questions}")
    print(f"  Avg Precision: {baseline_total_precision/total_questions:.2f}")
    print(f"  Avg Recall: {baseline_total_recall/total_questions:.2f}")

    print(f"\nBaseline + Search Results:")
    print(f"  Perfect matches: {search_perfect}/{total_questions}")
    print(f"  Avg Precision: {search_total_precision/total_questions:.2f}")
    print(f"  Avg Recall: {search_total_recall/total_questions:.2f}")

    print(f"\nImprovement: {search_perfect - baseline_perfect} more perfect matches")
    print("=" * 100)
