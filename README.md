# Schema Search

An MCP Server for Natural Language Search over Database Schemas. Find exact tables you need, with all their relationships mapped out, in milliseconds. No vector database setup required.

## Why

You have 200 tables in your database. Someone asks "where are user refunds stored?"

You could:
- Grep through SQL files for 20 minutes
- Pass the full schema to an LLM and watch it struggle with 200 tables

Or build schematic embeddings of your tables, store in-memory, and search in natural language.

### Benefits
- No vector database setup required
- Small memory footprint -- easily scales up to 1000 tables and 10,000 columns.
- Millisecond query latency

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

# Override search strategy
results = search.search("user_table", search_type="fuzzy")
```

## Configuration

Edit `config.yml`:

```yaml
embedding:
  location: "memory"
  model: "multi-qa-MiniLM-L6-cos-v1"
  metric: "cosine"

chunking:
  strategy: "markdown"  # or "llm"

search:
  strategy: "semantic"  # "semantic", "bm25", "fuzzy", or "hybrid"
  initial_top_k: 20
  rerank_top_k: 5
  semantic_weight: 0.67  # For hybrid search (fuzzy_weight = 1 - semantic_weight)

reranker:
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Set to null to disable reranking
```

## Search Strategies

Schema Search supports four search strategies:

- **semantic**: Embedding-based similarity search using sentence transformers
- **bm25**: Lexical search using BM25 ranking algorithm
- **fuzzy**: String matching on table/column names using fuzzy matching
- **hybrid**: Combines semantic and fuzzy scores (default: 67% semantic, 33% fuzzy)

Each strategy performs its own initial ranking, then optionally applies CrossEncoder reranking if `reranker.model` is configured. Set `reranker.model` to `null` to disable reranking.

## Performance Comparison

![Strategy Comparison](img/strategy_comparison.png)

Tested on a database with 26 tables using the sample `config.yml`:

### With Reranker (`Alibaba-NLP/gte-reranker-modernbert-base`)

- Reranking adds ~500ms latency but significantly improves accuracy
- BM25 struggles with semantic queries (40/50)

### Without Reranker** (set `reranker.model: null`):
- Semantic is most accurate (44/50) and fastest (26ms)
- Hybrid trades some accuracy for fuzzy matching capability (38/50)
- 10-30x faster but less accurate overall

You can override the search strategy at query time:

```python
# Use fuzzy search instead of default
results = search.search("user_table", search_type="fuzzy")

# Use BM25 for keyword-based search
results = search.search("transactions payments", search_type="bm25")

# Use hybrid for best of both worlds
results = search.search("where are user refunds?", search_type="hybrid")
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

1. **Extract schemas** from database using SQLAlchemy inspector
2. **Chunk schemas** into digestible pieces (markdown or LLM-generated summaries)
3. **Initial search** using selected strategy (semantic/BM25/fuzzy)
4. **Optional reranking** with CrossEncoder to refine results
5. **Expand via foreign keys** to find related tables (configurable hops)
6. Return top tables with full schema and relationships

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
      "args": ["postgresql://user:pass@localhost/db" "llm_api_key" "llm_base_url" "config.yml path"]
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
