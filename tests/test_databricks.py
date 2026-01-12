import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from schema_search import SchemaSearch


@pytest.fixture(scope="module")
def databricks_url():
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)

    host = os.getenv("DATABRICKS_HOST")
    http_path = os.getenv("DATABRICKS_HTTP_PATH")
    token = os.getenv("DATABRICKS_TOKEN")
    catalog = os.getenv("DATABRICKS_CATALOG")
    schema = os.getenv("DATABRICKS_SCHEMA")

    if not all([host, http_path, token, catalog]):
        pytest.skip(
            "DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN, or DATABRICKS_CATALOG not set in tests/.env"
        )

    url = f"databricks://token:{token}@{host}?http_path={http_path}&catalog={catalog}"
    if schema:
        url += f"&schema={schema}"

    return url


@pytest.fixture(scope="module")
def databricks_engine(databricks_url):
    return create_engine(
        databricks_url, connect_args={"user_agent_entry": "schema-search"}
    )


@pytest.mark.timeout(60)
def test_databricks_basic_query(databricks_engine):
    """Test basic Databricks connectivity."""
    print("\nTesting basic Databricks query...")
    print(f"Engine URL: {databricks_engine.url}")
    print("Attempting to connect...")

    try:
        with databricks_engine.connect() as conn:
            print("Connection established, executing query...")
            result = conn.execute(text("SELECT 1 as test"))
            print("Query executed, fetching result...")
            row = result.fetchone()
            assert row[0] == 1
        print("✓ Basic query works")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_databricks_list_tables(databricks_engine):
    """Test listing tables from Databricks catalog."""
    catalog = databricks_engine.url.query["catalog"]
    schema = databricks_engine.url.query.get("schema")

    if schema:
        print(f"\nListing tables from catalog: {catalog}, schema: {schema}")
    else:
        print(f"\nListing tables from catalog: {catalog}")

    query = text("""
        SELECT table_name, table_schema
        FROM system.information_schema.tables
        WHERE table_catalog = :catalog
        AND table_schema NOT IN ('information_schema', 'sys')
        LIMIT 5
    """)

    with databricks_engine.connect() as conn:
        result = conn.execute(query, {"catalog": catalog})
        rows = list(result)

    print(f"✓ Found {len(rows)} tables:")
    for row in rows:
        print(f"  - {row[1]}.{row[0]}")

    assert len(rows) > 0, "No tables found in catalog"


def test_databricks_connection(databricks_engine):
    """Test full SchemaSearch indexing and search."""
    print("\nTesting SchemaSearch with Databricks...")
    search = SchemaSearch(databricks_engine)

    print("Indexing...")
    search.index(force=True)
    print(f"✓ Indexed {len(search.schemas)} tables")

    print("Searching...")
    results = search.search("user")
    print(f"✓ Search complete: found {len(results.results)} results")

    assert len(results.results) > 0
