#!/usr/bin/env python3
import logging
from typing import Optional

from fastmcp import FastMCP
from sqlalchemy import create_engine

from schema_search import SchemaSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("schema-search")


@mcp.tool()
def schema_search(
    query: str,
    limit: int = 5,
) -> dict:
    """Search database schema using natural language.

    Finds relevant database tables and their relationships by searching through schema metadata
    using semantic similarity. Expands results by traversing foreign key relationships.

    Args:
        query: Natural language question about database schema (e.g., 'tables related to payments')
        limit: Maximum number of table schemas to return in results. Default: 5; Max: 10.

    Returns:
        Dictionary with 'results' (list of table schemas with columns, types, constraints, and relationships) and 'latency_sec' (query execution time)
    """
    limit = min(limit, 10)
    search_result = mcp.search_engine.search(query, limit=limit)  # type: ignore
    return {
        "results": search_result["results"],
        "latency_sec": search_result["latency_sec"],
    }


def run_server(
    database_url: str,
    config_path: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
):
    engine = create_engine(database_url)

    mcp.search_engine = SchemaSearch(  # type: ignore
        engine,
        config_path=config_path,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    )

    logger.info("Indexing database schema...")
    mcp.search_engine.index()  # type: ignore
    logger.info("Index ready")

    mcp.run()


def main():
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: schema-search <database_url> [config_path] [llm_api_key] [llm_base_url]"
        )
        sys.exit(1)

    database_url = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None
    llm_api_key = sys.argv[3] if len(sys.argv) > 3 else None
    llm_base_url = sys.argv[4] if len(sys.argv) > 4 else None

    run_server(database_url, config_path, llm_api_key, llm_base_url)


if __name__ == "__main__":
    main()
