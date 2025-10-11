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
def schema_search(query: str, hops: int = 1, limit: int = 5) -> dict:
    """Search database schema using natural language.

    Args:
        query: Natural language question about database schema (e.g., 'where are user refunds stored?')
        hops: Number of foreign key hops for graph expansion. Recommended: 1 or 0
        limit: Maximum number of table schemas to return

    Returns:
        Dictionary containing search results (schema of the tables) and latency information
    """
    search_result = mcp.search_engine.search(query, hops=hops, limit=limit)  # type: ignore
    return {
        "results": search_result["results"],
        "latency_sec": search_result["latency_sec"],
    }


def main(
    database_url: str,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    config_path: Optional[str] = None,
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m schema_search.mcp_server <database_url> [llm_api_key] [llm_base_url] [config_path]"
        )
        sys.exit(1)

    database_url = sys.argv[1]
    llm_api_key = sys.argv[2] if len(sys.argv) > 2 else None
    llm_base_url = sys.argv[3] if len(sys.argv) > 3 else None
    config_path = sys.argv[4] if len(sys.argv) > 4 else None

    main(database_url, llm_api_key, llm_base_url, config_path)
