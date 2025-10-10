from typing import Dict

from schema_search.chunkers.base import BaseChunker
from schema_search.chunkers.markdown import MarkdownChunker
from schema_search.chunkers.llm import LLMChunker


def create_chunker(config: Dict) -> BaseChunker:
    chunking_config = config["chunking"]
    if chunking_config["strategy"] == "llm":
        return LLMChunker(
            max_tokens=chunking_config["max_tokens"],
            overlap_tokens=chunking_config["overlap_tokens"],
            model=chunking_config["model"],
        )
    else:
        return MarkdownChunker(
            max_tokens=chunking_config["max_tokens"],
            overlap_tokens=chunking_config["overlap_tokens"],
        )
