import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine

from schema_search import SchemaSearch


@pytest.fixture(scope="module")
def databricks_url():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")
    catalog = os.getenv("DATABRICKS_CATALOG")

    if not all([host, http_path, token, catalog]):
        pytest.skip(
            "DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN, or DATABRICKS_CATALOG not set in tests/.env"
        )

    return f"databricks://token:{token}@{host}:443/{catalog}?http_path={http_path}"


def test_databricks_connection(databricks_url):
    engine = create_engine(databricks_url)
    search = SchemaSearch(engine)

    search.index(force=True)

    print(search.schemas)
    results = search.search("user")

    assert len(results["results"]) > 0
    print(f"\n✓ Databricks test passed: found {len(results['results'])} results")
