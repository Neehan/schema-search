# Schema Search MCP Server

Ask questions about your database in natural language. Get back the exact tables you need, with all their relationships mapped out.

## Why

Suppose you have 200 tables in a database. Someone asks "where are user refunds stored?" 

You could:
- Grep through SQL files for 20 minutes
- Ask an LLM, which will struggle to sift through 200 table schemas

Or just ask the database directly.

## Install

```bash
# PostgreSQL
pip install schema-search[postgres,mcp]

# MySQL
pip install schema-search[mysql,mcp]

# Snowflake
pip install schema-search[snowflake,mcp]

# BigQuery
pip install schema-search[bigquery,mcp]
```

## Use

```python
from sqlalchemy import create_engine
from schema_search import SchemaSearch

engine = create_engine("postgresql://user:pass@localhost/db")
search = SchemaSearch(engine)

search.index()
results = search.search("where are user refunds stored?")

for result in results['results']:
    print(result['table'])           # "refund_transactions"
    print(result['schema'])           # Full column info, types, constraints
    print(result['related_tables'])   # ["users", "payments", "transactions"]
```

## Configuration

Edit `config.yml`:

```yaml
embedding:
  location: "memory"  # vectordb coming soon
  model: "multi-qa-MiniLM-L6-cos-v1"
  metric: "cosine"

chunking:
  strategy: "raw"  # or "llm"

reranker:
  strategy: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

### LLM Chunking

Use LLM to generate semantic summaries instead of raw schema text:

1. Set `strategy: "llm"` in `config.yml`
2. Pass API credentials:

```python
search = SchemaSearch(
    engine,
    llm_api_key="sk-ant-...",
    llm_base_url="https://api.anthropic.com"  # optional
)
```

## How It Works

1. Semantic search on schema chunks (in-memory embeddings)
2. Expand results via foreign key graph (N-hops)
3. Re-rank with CrossEncoder
4. Return top tables with relationships

Cache stored in `.schema_search_cache/`.

## Performance

Tested on a realistic database with 25 tables and 200+ columns. Average query latency: **<40ms**.

## MCP Server

Integrate with Claude Desktop or any MCP client.

### Setup

Add to your MCP config (e.g., `~/.cursor/mcp.json` or Claude Desktop config):

```json
{
  "mcpServers": {
    "schema-search": {
      "command": "schema-search-mcp",
      "args": ["postgresql://user:pass@localhost/db"]
    }
  }
}
```

### CLI Usage

```bash
schema-search-mcp "postgresql://user:pass@localhost/db"
```

Optional args: `[llm_api_key] [llm_base_url] [config_path]`

The server exposes `schema_search(query, hops, limit)` for natural language schema queries.

## License

MIT
