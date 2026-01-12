import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import sys

# Load environment
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

host = os.getenv("DATABRICKS_HOST")
http_path = os.getenv("DATABRICKS_HTTP_PATH")
token = os.getenv("DATABRICKS_TOKEN")
catalog = os.getenv("DATABRICKS_CATALOG")
schema = os.getenv("DATABRICKS_SCHEMA")

print(f"Host: {host}")
print(f"HTTP Path: {http_path}")
print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print()

url = f"databricks://token:{token}@{host}?http_path={http_path}&catalog={catalog}&schema={schema}"

print("Creating engine...")
engine = create_engine(
    url,
    connect_args={
        "user_agent_entry": "schema-search",
        "_retry_stop_after_attempts_count": 1,  # Disable retries
        "_socket_timeout": 10  # 10 second timeout
    }
)
print(f"Engine created: {engine.url}")
print()

print("Attempting to connect...")
sys.stdout.flush()

try:
    with engine.connect() as conn:
        print("Connected! Executing query...")
        sys.stdout.flush()

        result = conn.execute(text("SELECT 1 as test"))
        print("Query executed, fetching result...")
        sys.stdout.flush()

        row = result.fetchone()
        print(f"Success! Result: {row}")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
