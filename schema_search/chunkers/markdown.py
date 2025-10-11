from schema_search.chunkers.base import BaseChunker
from schema_search.types import TableSchema


class MarkdownChunker(BaseChunker):
    def _generate_content(self, table_name: str, schema: TableSchema) -> str:
        lines = [f"Table: {table_name}"]

        if schema["primary_keys"]:
            lines.append(f"Primary keys: {', '.join(schema['primary_keys'])}")

        if schema["columns"]:
            col_names = [col["name"] for col in schema["columns"]]
            lines.append(f"Columns: {', '.join(col_names)}")

        if schema["foreign_keys"]:
            related = [fk["referred_table"] for fk in schema["foreign_keys"]]
            lines.append(f"Related to: {', '.join(related)}")

        if schema["indices"]:
            idx_names = [idx["name"] for idx in schema["indices"] if idx["name"]]
            if idx_names:
                lines.append(f"Indexes: {', '.join(idx_names)}")

        return "\n".join(lines)
