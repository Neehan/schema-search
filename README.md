# Schema Search

An MCP Server for Natural Language Search over RDBMS Schemas. Find exact tables you need, with all their relationships mapped out, in milliseconds. No vector database setup is required.

## Why

You have 200 tables in your database. Someone asks "where are user refunds stored?"

You could:
- Grep through SQL files for 20 minutes
- Pass the full schema to an LLM and watch it struggle with 200 tables

Or **build schematic embeddings of your tables, store in-memory, and query in natural language in an MCP server**.

### Benefits
- No vector database setup is required
- Small memory footprint -- easily scales up to 1000 tables and 10,000+ columns.
- Millisecond query latency

## Install

```bash
# With uv - PostgreSQL (recommended)
uv pip install "schema-search[postgres,mcp]"

# With pip - PostgreSQL
pip install "schema-search[postgres,mcp]"

# Other databases
uv pip install "schema-search[mysql,mcp]"      # MySQL
uv pip install "schema-search[snowflake,mcp]"  # Snowflake
uv pip install "schema-search[bigquery,mcp]"   # BigQuery
```

## MCP Server

Integrate with Claude Desktop or any MCP client.

### Setup

Add to your MCP config (e.g., `~/.cursor/mcp.json` or Claude Desktop config):

**Using uv (Recommended):**
```json
{
  "mcpServers": {
    "schema-search": {
      "command": "uvx",
      "args": ["schema-search[postgres,mcp]", "postgresql://user:pass@localhost/db", "optional config.yml path", "optional llm_api_key", "optional llm_base_url"]
    }
  }
}
```

**Using pip:**
```json
{
  "mcpServers": {
    "schema-search": {
      "command": "path/to/schema-search", // conda: /Users/<username>/opt/miniconda3/envs/<your env>/bin/schema-search",
      "args": ["postgresql://user:pass@localhost/db", "optional config.yml path", "optional llm_api_key", "optional llm_base_url"]
    }
  }
}
```


The LLM API key and base url are only required if you use LLM-generated schema summaries (`config.chunking.strategy = 'llm'`).

### CLI Usage

```bash
schema-search "postgresql://user:pass@localhost/db"
```

Optional args: `[config_path] [llm_api_key] [llm_base_url]`

The server exposes `schema_search(query, hops, limit)` for natural language schema queries.

## Python Use

```python
from sqlalchemy import create_engine
from schema_search import SchemaSearch

engine = create_engine("postgresql://user:pass@localhost/db")
search = SchemaSearch(engine)

search.index(force=False) # default is False
results = search.search("where are user refunds stored?")

for result in results['results']:
    print(result['table'])           # "refund_transactions"
    print(result['schema'])           # Full column info, types, constraints
    print(result['related_tables'])   # ["users", "payments", "transactions"]

# Override hops, limit, search strategy
results = search.search("user_table", hops=0, limit=5, search_type="semantic")

```

`SchemaSearch.index()` automatically detects schema changes and refreshes cached metadata, so you rarely need to force a reindex manually.

## Configuration

Edit `[config.yml](config.yml)`:

```yaml
logging:
  level: "WARNING"

embedding:
  location: "memory" # Options: "memory", "vectordb" (coming soon)
  model: "multi-qa-MiniLM-L6-cos-v1"
  metric: "cosine" # Options: "cosine", "euclidean", "manhattan", "dot"
  batch_size: 32
  show_progress: false
  cache_dir: "/tmp/.schema_search_cache"

chunking:
  strategy: "raw" # Options: "raw", "llm"
  max_tokens: 256
  overlap_tokens: 50
  model: "gpt-4o-mini"

search:
  # Search strategy: "semantic" (embeddings), "bm25" (BM25 lexical), "fuzzy" (fuzzy string matching), "hybrid" (semantic + bm25)
  strategy: "hybrid"
  initial_top_k: 20
  rerank_top_k: 5
  semantic_weight: 0.67 # For hybrid search (bm25_weight = 1 - semantic_weight)
  hops: 1 # Number of foreign key hops for graph expansion (0-2 recommended)

reranker:
  # CrossEncoder model for reranking. Set to null to disable reranking
  model: null # "Alibaba-NLP/gte-reranker-modernbert-base"

schema:
  include_columns: true
  include_indices: true
  include_foreign_keys: true
  include_constraints: true
```

## Search Strategies

Schema Search supports four search strategies:

- **semantic**: Embedding-based similarity search using sentence transformers
- **bm25**: Lexical search using BM25 ranking algorithm
- **fuzzy**: String matching on table/column names using fuzzy matching
- **hybrid**: Combines semantic and bm25 scores (default: 67% semantic, 33% fuzzy)

Each strategy performs its own initial ranking, then optionally applies CrossEncoder reranking if `reranker.model` is configured. Set `reranker.model` to `null` to disable reranking.

## Performance Comparison
We [benchmarked](/tests/test_spider_eval.py) on the Spider dataset (1,234 train queries across 18 databases) using the default `config.yml`.  

**Memory:** The embedding model requires ~90 MB and the optional reranker adds ~155 MB. Actual process memory depends on your Python runtime.

### Without Reranker (`reranker.model: null`)
![Without Reranker](img/spider_benchmark_without_reranker.png)
- **Indexing:** 0.22s ± 0.08s per database (18 total).
- **Accuracy:** Hybrid leads with Recall@1 62% / MRR 0.93; Semantic follows at Recall@1 58% / MRR 0.89.
- **Latency:** BM25 and Fuzzy return in ~5ms; Semantic spends ~15ms; Hybrid (semantic + fuzzy) averages 52ms.
- **Fuzzy baseline:** Recall@1 22%, highlighting the need for semantic signals on natural-language queries.

### With Reranker (`Alibaba-NLP/gte-reranker-modernbert-base`)
![With Reranker](img/spider_benchmark_with_reranker.png)
- **Indexing:** 0.25s ± 0.05s per database (same 18 DBs).
- **Accuracy:** All strategies converge around Recall@1 62% and MRR ≈ 0.92; Fuzzy jumps from 51% → 92% MRR.
- **Latency trade-off:** Extra CrossEncoder pass lifts per-query latency to ~0.18–0.29s depending on strategy.
- **Recommendation:** Enable the reranker when accuracy matters most; disable it for ultra-low-latency lookups.


You can override the search strategy, hops, and limit at query time:

```python
# Use fuzzy search instead of default
results = search.search("user_table", search_type="fuzzy")

# Use BM25 for keyword-based search
results = search.search("transactions payments", search_type="bm25")

# Use hybrid for best of both worlds
results = search.search("where are user refunds?", search_type="hybrid")

# Override hops and limit
results = search.search("user refunds", hops=2, limit=10)  # Expand 2 hops, return 10 tables

# Disable graph expansion
results = search.search("user_table", hops=0)  # Only direct matches, no foreign key traversal
```

### LLM Chunking

Use LLM to generate semantic summaries instead of raw schema text:

1. Set `strategy: "llm"` in `config.yml`
2. Pass API credentials:

```python
search = SchemaSearch(
    engine,
    llm_api_key="sk-...",
    llm_base_url="https://api.openai.com/v1/"  # optional
)
```

## How It Works

1. **Extract schemas** from database using SQLAlchemy inspector
2. **Chunk schemas** into digestible pieces (markdown or LLM-generated summaries)
3. **Initial search** using selected strategy (semantic/BM25/fuzzy)
4. **Expand via foreign keys** to find related tables (configurable hops)
5. **Optional reranking** with CrossEncoder to refine results
6. Return top tables with full schema and relationships

Cache stored in `.schema_search_cache/` (configurable in `config.yml`)

## License

MIT
