import os
import json
import logging

from anthropic import Anthropic
from dotenv import load_dotenv

from schema_search.chunkers.base import BaseChunker
from schema_search.types import TableSchema

load_dotenv()

logger = logging.getLogger(__name__)


class LLMChunker(BaseChunker):
    def __init__(self, max_tokens: int, overlap_tokens: int, model: str):
        super().__init__(max_tokens, overlap_tokens)
        self.model = model
        self.llm_client = Anthropic(
            api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_BASE_URL")
        )
        logger.info(f"Schema Summarizer Model: {self.model}")

    def _generate_content(self, table_name: str, schema: TableSchema) -> str:
        prompt = f"""Generate a concise 250 tokens or less semantic summary of this database table schema. Focus on:
1. What entity or concept this table represents
2. Key data it stores (main columns)
3. How it relates to other tables
4. Any important constraints or indices

Keep it brief and semantic, optimized for embedding-based search.

Schema:
{json.dumps(schema, indent=2)}

Return ONLY the summary text, no preamble."""

        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.content[0].text.strip()  # type: ignore
        logger.debug(f"Generated LLM summary for {table_name}: {summary[:100]}...")

        return f"Table: {table_name}\n{summary}"
