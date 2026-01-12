# type: ignore
"""Spider benchmark evaluation.

Warning: this test intentionally creates and drops PostgreSQL databases
matching Spider db_ids. Only run against an isolated server you are
comfortable wiping.
"""

import re
import time
import json
from collections import defaultdict
from typing import List, Set, Dict, Any, TypedDict
from pathlib import Path

import numpy as np
import pytest
from datasets import load_dataset
from sqlalchemy import create_engine, text
from tqdm import tqdm
from schema_search import SchemaSearch
from schema_search.types import SearchType


class StrategyStats(TypedDict):
    recall_at_1: List[float]
    recall_at_3: List[float]
    recall_at_5: List[float]
    mrr: List[float]
    precision_at_1: List[float]
    precision_at_3: List[float]
    precision_at_5: List[float]
    latency: List[float]
    num_queries: int


# WARNING: Do not swap this for an environment variable—this benchmark is meant
#          to run against an explicit throwaway Postgres instance.
DATABASE_URL = "postgresql://user:pass@localhost/db"
MIN_TABLES = 10


def parse_spider_schema(schema_str: str) -> Dict[str, List[tuple]]:
    """Parse spider-schema format into table definitions."""
    tables = {}
    table_parts = schema_str.split(" | ")

    for table_part in table_parts:
        if " : " not in table_part:
            continue

        table_name, columns_str = table_part.split(" : ", 1)
        table_name = table_name.strip()

        columns = []
        col_parts = columns_str.split(" , ")

        for col_part in col_parts:
            match = re.match(r"(.+?)\s*\((\w+)\)", col_part.strip())
            if match:
                col_name = match.group(1).strip()
                col_type = match.group(2).strip()
                sql_type = "INTEGER" if col_type == "number" else "TEXT"
                columns.append((col_name, sql_type))

        if columns:
            tables[table_name] = columns

    return tables


def extract_tables_from_sql(sql: str) -> Set[str]:
    """Extract table names from SQL query using regex.

    Handles: aliases, quoted names, schemas, comma-separated joins.
    """
    sql_lower = sql.lower()

    def clean_table_name(name: str) -> str:
        """Strip quotes, brackets, schema prefix, and aliases."""
        name = name.strip().strip('`"[]')
        if "." in name:
            name = name.split(".")[-1]
        if " " in name:
            name = name.split()[0]
        return name.strip()

    tables = set()

    from_pattern = r"\bfrom\s+([`\"[]?[\w]+[`\"\]]?(?:\.[\w]+)?(?:\s+(?:as\s+)?\w+)?)"
    join_pattern = r"\bjoin\s+([`\"[]?[\w]+[`\"\]]?(?:\.[\w]+)?(?:\s+(?:as\s+)?\w+)?)"

    for match in re.finditer(from_pattern, sql_lower):
        table = clean_table_name(match.group(1))
        if table:
            tables.add(table)

    for match in re.finditer(join_pattern, sql_lower):
        table = clean_table_name(match.group(1))
        if table:
            tables.add(table)

    from_clause_match = re.search(
        r"\bfrom\s+([^;]+?)(?:\bwhere\b|\bjoin\b|\bgroup\b|\border\b|\blimit\b|$)",
        sql_lower,
    )
    if from_clause_match:
        from_clause = from_clause_match.group(1)
        for table_expr in re.split(r"\s*,\s*", from_clause):
            table = clean_table_name(table_expr)
            if table and re.match(r"^[a-zA-Z_][\w]*$", table):
                tables.add(table)

    return tables


def create_database_from_schema(db_id: str, schema_str: str):
    """Create PostgreSQL database from spider-schema string.

    Returns:
        tuple: (db_engine, num_tables)
    """
    safe_db_id = db_id.lower().replace("-", "_")

    assert (
        "localhost" in DATABASE_URL
    ), "DATABASE_URL must be a test database on localhost"

    admin_engine = create_engine(
        DATABASE_URL, isolation_level="AUTOCOMMIT", pool_pre_ping=True, pool_recycle=60
    )

    with admin_engine.connect() as conn:
        conn.execute(
            text(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{safe_db_id}' AND pid != pg_backend_pid()"
            )
        )
        conn.execute(text(f"DROP DATABASE IF EXISTS {safe_db_id}"))
        conn.execute(text(f"CREATE DATABASE {safe_db_id}"))

    admin_engine.dispose()

    db_engine = create_engine(
        f"postgresql://user:pass@localhost/{safe_db_id}",
        pool_pre_ping=True,
        pool_recycle=60,
    )
    tables = parse_spider_schema(schema_str)

    with db_engine.connect() as conn:
        for table_name, columns in tables.items():
            col_defs = ", ".join(
                [f'"{col_name}" {col_type}' for col_name, col_type in columns]
            )
            create_sql = f'CREATE TABLE "{table_name}" ({col_defs})'
            conn.execute(text(create_sql))
        conn.commit()

    return db_engine, len(tables)


def calculate_recall_at_k(
    predicted: List[str], ground_truth: Set[str], k: int
) -> float:
    """Calculate Recall@k = (relevant retrieved) / (total relevant).

    For k=1, ground_truth={A,B,C}, predicted=[A,...]: Recall@1 = 1/3 = 0.33
    """
    if not ground_truth:
        return 0.0

    predicted_at_k = set([p.lower() for p in predicted[:k]])
    ground_truth_lower = set([g.lower() for g in ground_truth])

    matches = len(predicted_at_k & ground_truth_lower)
    return matches / len(ground_truth_lower)


def calculate_precision_at_k(
    predicted: List[str], ground_truth: Set[str], k: int
) -> float:
    """Calculate Precision@k = (relevant retrieved) / k.

    For k=1, ground_truth={A,B,C}, predicted=[A,...]: Precision@1 = 1/1 = 1.0
    """
    if k == 0:
        return 0.0

    predicted_at_k = set([p.lower() for p in predicted[:k]])
    ground_truth_lower = set([g.lower() for g in ground_truth])

    matches = len(predicted_at_k & ground_truth_lower)
    return matches / k


def calculate_mrr(predicted: List[str], ground_truth: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank = 1/rank of first correct item."""
    if not ground_truth:
        return 0.0

    ground_truth_lower = set([g.lower() for g in ground_truth])

    for rank, pred in enumerate(predicted, 1):
        if pred.lower() in ground_truth_lower:
            return 1.0 / rank

    return 0.0


def save_benchmark_results(results_by_strategy, index_latencies, strategies):
    """Save benchmark results as JSON."""
    import yaml

    config_path = Path(__file__).parent.parent / "config.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    has_reranker = config.get("reranker", {}).get("model") is not None
    reranker_model = config.get("reranker", {}).get("model")

    img_dir = Path(__file__).parent.parent / "img"
    img_dir.mkdir(exist_ok=True)

    output = {
        "reranker_enabled": has_reranker,
        "reranker_model": reranker_model,
        "indexing": {},
        "strategies": {},
    }

    if index_latencies:
        output["indexing"] = {
            "num_databases": len(index_latencies),
            "mean_latency": float(np.mean(index_latencies)),
            "std_latency": float(np.std(index_latencies)),
        }

    for strategy in strategies:
        stats = results_by_strategy[strategy]
        if stats["num_queries"] == 0:
            continue

        strategy_results = {"num_queries": stats["num_queries"]}

        metric_names = [
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "latency",
        ]

        for metric_name in metric_names:
            if metric_name in stats and stats[metric_name]:
                values = np.array(stats[metric_name])
                strategy_results[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        output["strategies"][strategy] = strategy_results

    output_filename = (
        "spider_benchmark_with_reranker.json"
        if has_reranker
        else "spider_benchmark_without_reranker.json"
    )
    output_path = img_dir / output_filename

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nBenchmark results saved to: {output_path}")


def cleanup_spider_databases():
    """Drop Spider test databases and clear cache before starting tests.

    Only drops databases matching Spider naming pattern (alphanumeric + underscores).
    """
    import shutil
    from pathlib import Path

    admin_engine = create_engine(DATABASE_URL, isolation_level="AUTOCOMMIT")

    with admin_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT datname FROM pg_database WHERE datname NOT IN ('postgres', 'template0', 'template1', 'db')"
            )
        )
        databases = [row[0] for row in result]

        for db_name in databases:
            if re.match(r"^[a-z0-9_]+$", db_name):
                conn.execute(
                    text(
                        f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db_name}'"
                    )
                )
                conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))

    admin_engine.dispose()

    cache_path = Path("/tmp/.schema_search_cache")
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"Cleared cache: {cache_path}")


@pytest.fixture(scope="module")
def spider_data():
    """Load Spider questions/queries and spider-schema definitions."""
    cleanup_spider_databases()

    spider = load_dataset("spider", split="train")
    spider_schema = load_dataset("richardr1126/spider-schema", split="train")

    schema_map = {ex["db_id"]: ex for ex in spider_schema}

    return spider, schema_map


def test_spider_evaluation(spider_data):
    """Evaluate schema search on Spider benchmark.

    Tests table-level retrieval accuracy across multiple databases.
    Metrics: Recall@1, Recall@3, Recall@5, MRR, Precision@k, Latency
    """
    spider, schema_map = spider_data

    results_by_strategy: Dict[str, StrategyStats] = defaultdict(
        lambda: StrategyStats(
            recall_at_1=[],
            recall_at_3=[],
            recall_at_5=[],
            mrr=[],
            precision_at_1=[],
            precision_at_3=[],
            precision_at_5=[],
            latency=[],
            num_queries=0,
        )
    )

    strategies: List[SearchType] = ["hybrid", "fuzzy", "bm25", "semantic"]

    current_db_id = None
    search_engine = None
    db_engine = None
    index_latencies = []
    num_dbs_filtered = 0
    total_queries_evaluated = 0

    for example in tqdm(spider, desc="Evaluating Spider"):
        db_id = example["db_id"]
        question = example["question"]
        sql = example["query"]

        if db_id not in schema_map:
            continue

        if db_id != current_db_id:
            if db_engine is not None:
                db_engine.dispose()

            schema_str = schema_map[db_id]["Schema (values (type))"]
            db_engine, num_tables = create_database_from_schema(db_id, schema_str)

            if num_tables < MIN_TABLES:
                num_dbs_filtered += 1
                safe_db_id = db_id.lower().replace("-", "_")
                db_engine.dispose()

                admin_engine = create_engine(DATABASE_URL, isolation_level="AUTOCOMMIT")
                with admin_engine.connect() as conn:
                    conn.execute(
                        text(
                            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{safe_db_id}'"
                        )
                    )
                    conn.execute(text(f"DROP DATABASE IF EXISTS {safe_db_id}"))
                admin_engine.dispose()

                db_engine = None
                search_engine = None
                current_db_id = db_id
                continue

            search_engine = SchemaSearch(db_engine)

            index_start = time.time()
            search_engine.index(force=False)
            index_latency = time.time() - index_start
            index_latencies.append(index_latency)

            current_db_id = db_id

        if search_engine is None:
            continue

        ground_truth_tables = extract_tables_from_sql(sql)

        if not ground_truth_tables:
            continue

        for strategy in strategies:
            start_time = time.time()
            response = search_engine.search(
                question, search_type=strategy, limit=10, hops=1
            )
            latency = time.time() - start_time

            predicted_tables = [r["table"] for r in response.results]

            for k in [1, 3, 5]:
                results_by_strategy[strategy][f"recall_at_{k}"].append(
                    calculate_recall_at_k(predicted_tables, ground_truth_tables, k)
                )
                results_by_strategy[strategy][f"precision_at_{k}"].append(
                    calculate_precision_at_k(predicted_tables, ground_truth_tables, k)
                )

            results_by_strategy[strategy]["mrr"].append(
                calculate_mrr(predicted_tables, ground_truth_tables)
            )
            results_by_strategy[strategy]["latency"].append(latency)
            # Type-safe increment
            current_count: int = results_by_strategy[strategy]["num_queries"]  # type: ignore
            results_by_strategy[strategy]["num_queries"] = current_count + 1

        total_queries_evaluated += 1

    print(f"\nFiltered out {num_dbs_filtered} databases with <{MIN_TABLES} tables")
    print(
        f"Evaluated {total_queries_evaluated} queries on {len(index_latencies)} databases"
    )

    if db_engine is not None:
        db_engine.dispose()

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")

    if index_latencies:
        mean_index = np.mean(index_latencies)
        std_index = np.std(index_latencies)
        print(f"\nINDEXING")
        print(f"  Databases indexed: {len(index_latencies)}")
        print(f"  Index latency: {mean_index:.3f}s ± {std_index:.3f}s")

    for strategy in strategies:
        stats = results_by_strategy[strategy]

        if stats["num_queries"] == 0:
            continue

        print(f"\n{strategy.upper()}")
        print(f"  Queries: {stats['num_queries']}")

        metric_order = [
            "recall_at_1",
            "recall_at_3",
            "recall_at_5",
            "mrr",
            "precision_at_1",
            "precision_at_3",
            "precision_at_5",
            "latency",
        ]
        metric_labels = {
            "recall_at_1": "Recall@1",
            "recall_at_3": "Recall@3",
            "recall_at_5": "Recall@5",
            "mrr": "MRR",
            "precision_at_1": "Precision@1",
            "precision_at_3": "Precision@3",
            "precision_at_5": "Precision@5",
            "latency": "Latency",
        }

        for metric_name in metric_order:
            if metric_name in stats and stats[metric_name]:
                values = np.array(stats[metric_name])
                mean = np.mean(values)
                std = np.std(values)
                label = metric_labels[metric_name]

                if metric_name == "latency":
                    print(f"  {label}: {mean:.3f}s ± {std:.3f}s")
                else:
                    print(f"  {label}: {mean:.3f} ± {std:.3f}")

        mean_recall_1 = np.mean(stats["recall_at_1"]) if stats["recall_at_1"] else 0.0
        mean_mrr = np.mean(stats["mrr"]) if stats["mrr"] else 0.0
        assert mean_recall_1 >= 0.0, f"{strategy}: Invalid recall@1"
        assert mean_mrr >= 0.0, f"{strategy}: Invalid MRR"

    save_benchmark_results(results_by_strategy, index_latencies, strategies)
