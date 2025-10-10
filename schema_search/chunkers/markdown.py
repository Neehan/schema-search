from schema_search.chunkers.base import BaseChunker
from schema_search.types import TableSchema


class MarkdownChunker(BaseChunker):
    def _generate_content(self, table_name: str, schema: TableSchema) -> str:
        lines = [f"Table: {table_name}"]

        if schema["columns"]:
            col_names = [col["name"] for col in schema["columns"]]
            cols_per_line = 10
            for i in range(0, len(col_names), cols_per_line):
                batch = col_names[i : i + cols_per_line]
                col_names_str = ", ".join(batch)
                lines.append(f"Columns:{col_names_str}")

        if schema["foreign_keys"]:
            related = [fk["referred_table"] for fk in schema["foreign_keys"]]
            lines.append(f"Related to: {', '.join(related)}")

        if schema["indices"]:
            idx_names = [idx["name"] for idx in schema["indices"] if idx["name"]]
            if idx_names:
                lines.append(f"Indexes: {', '.join(idx_names)}")

        return "\n".join(lines)
