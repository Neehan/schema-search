import json
import logging
from typing import Optional, TYPE_CHECKING

from schema_search.chunkers.base import BaseChunker
from schema_search.types import TableSchema
from schema_search.utils.lazy_import import lazy_import_check

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMChunker(BaseChunker):
    def __init__(
        self,
        max_tokens: int,
        overlap_tokens: int,
        model: str,
        llm_api_key: Optional[str],
        llm_base_url: Optional[str],
        show_progress: bool = False,
    ):
        super().__init__(max_tokens, overlap_tokens, show_progress)
        self.model = model
        openai = lazy_import_check("openai", "llm", "LLM-based chunking")
        self.llm_client: "OpenAI" = openai.OpenAI(api_key=llm_api_key, base_url=llm_base_url)
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

        response = self.llm_client.chat.completions.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )

        summary = response.choices[0].message.content.strip()  # type: ignore
        logger.debug(f"Generated LLM summary for {table_name}: {summary[:100]}...")

        return f"Table: {table_name}\n{summary}"
