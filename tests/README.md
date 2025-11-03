# Integration Tests

## Setup

1. Install test dependencies with database driver:
   ```bash
   # For PostgreSQL (use quotes for zsh)
   pip install -e ".[test,postgres,semantic,llm]"
   
   ```

2. Create a `.env` file in the `tests/` directory:
   ```bash
   cp tests/env.template tests/.env
   ```

3. Edit `tests/.env` and set your database connection URL:
   ```
   DATABASE_URL=postgresql://user:password@localhost:5432/your_database
   
   # Optional: for LLM-based summarization
   LLM_API_KEY=your_openai_api_key
   LLM_BASE_URL=https://api.openai.com/v1
   ```

4. (Optional) Enable LLM summarization in `config.yml`:
   ```yaml
   chunking:
     strategy: "llm"
     model: "gpt-4o-mini"
   ```

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run with verbose output:
```bash
pytest tests/ -v
```

Run with print statements visible:
```bash
pytest tests/ -s
```

## Test Database

The integration tests require access to a real database with tables. Make sure:
- The database URL in `.env` is correct
- The database contains at least one table
- The user has read permissions on the schema

## Clean Cache

If you need to rebuild the index, delete the cache directory:
```bash
rm -rf .schema_search_cache/
```

