from typing import Dict, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from tqdm import tqdm

from schema_search.types import TableSchema


@dataclass
class Chunk:
    table_name: str
    content: str
    chunk_id: int
    token_count: int


class BaseChunker(ABC):
    def __init__(self, max_tokens: int, overlap_tokens: int):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_schemas(self, schemas: Dict[str, TableSchema]) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_id = 0

        for table_name, schema in tqdm(
            schemas.items(), desc="Chunking tables", unit="table"
        ):
            table_chunks = self._chunk_table(table_name, schema, chunk_id)
            chunks.extend(table_chunks)
            chunk_id += len(table_chunks)

        return chunks

    @abstractmethod
    def _generate_content(self, table_name: str, schema: TableSchema) -> str:
        pass

    def _chunk_table(
        self, table_name: str, schema: TableSchema, start_id: int
    ) -> List[Chunk]:
        content = self._generate_content(table_name, schema)
        lines = content.split("\n")

        header = f"Table: {table_name}"
        header_tokens = self._estimate_tokens(header)

        chunks: List[Chunk] = []
        current_chunk_lines = [header]
        current_tokens = header_tokens
        chunk_id = start_id

        for line in lines[1:]:
            line_tokens = self._estimate_tokens(line)

            if (
                current_tokens + line_tokens > self.max_tokens
                and len(current_chunk_lines) > 1
            ):
                chunk_content = "\n".join(current_chunk_lines)
                chunks.append(
                    Chunk(
                        table_name=table_name,
                        content=chunk_content,
                        chunk_id=chunk_id,
                        token_count=current_tokens,
                    )
                )
                chunk_id += 1

                current_chunk_lines = [header]
                current_tokens = header_tokens

            current_chunk_lines.append(line)
            current_tokens += line_tokens

        if len(current_chunk_lines) > 1:
            chunk_content = "\n".join(current_chunk_lines)
            chunks.append(
                Chunk(
                    table_name=table_name,
                    content=chunk_content,
                    chunk_id=chunk_id,
                    token_count=current_tokens,
                )
            )

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        return len(text.split()) + len(text) // 4
