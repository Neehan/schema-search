"""
Markdown renderer for search results.
"""

from typing import TYPE_CHECKING, List

from schema_search.renderers.base import BaseRenderer
from schema_search.types import TableSchema, ColumnInfo, SearchResultItem

if TYPE_CHECKING:
    from schema_search.types import SearchResult


class MarkdownRenderer(BaseRenderer):
    """Renders search results as formatted markdown."""

    def render(self, search_result: "SearchResult") -> str:
        """Render search results as markdown string."""
        lines = []

        # Header
        lines.append("# Schema Search Results")
        lines.append("")
        lines.append(f"**Query Latency**: {search_result.latency_sec:.3f} seconds")
        lines.append(f"**Results Found**: {len(search_result.results)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Individual results
        for idx, result in enumerate(search_result.results, 1):
            lines.append(self._render_result_item(result, idx))
            if idx < len(search_result.results):
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def _render_result_item(self, result: SearchResultItem, rank: int) -> str:
        """Render a single search result item."""
        lines = []

        # Result header with rank and score
        lines.append(f"## Result {rank}: {result['table']}")
        lines.append(f"**Score**: {result['score']:.4f}")
        lines.append("")

        # Related tables
        related_tables = result.get("related_tables")
        if related_tables:
            lines.append(f"**Related Tables**: {', '.join(related_tables)}")
            lines.append("")

        # Matched chunks
        matched_chunks = result.get("matched_chunks")
        if matched_chunks:
            lines.append("**Matched Content**:")
            for chunk in matched_chunks:
                lines.append("```")
                lines.append(chunk)
                lines.append("```")
            lines.append("")

        # Table schema
        lines.append(self._render_table_schema(result["schema"]))

        return "\n".join(lines)

    def _render_table_schema(self, schema: TableSchema) -> str:
        """Render table schema as markdown."""
        lines = []

        # Table name
        lines.append(f"### {schema['name']}")
        lines.append("")

        # Primary keys
        if schema.get("primary_keys"):
            lines.append(f"**Primary Keys**: {', '.join(schema['primary_keys'])}")
            lines.append("")

        # Columns
        columns = schema.get("columns")
        if columns:
            lines.append("**Columns**:")
            for col in columns:
                lines.append(self._render_column(col))
            lines.append("")

        # Foreign keys
        foreign_keys = schema.get("foreign_keys")
        if foreign_keys:
            lines.append("**Foreign Keys**:")
            for fk in foreign_keys:
                constrained = ", ".join(fk["constrained_columns"])
                referred = ", ".join(fk["referred_columns"])
                lines.append(f"  - {constrained} -> {fk['referred_table']}({referred})")
            lines.append("")

        # Indices
        indices = schema.get("indices")
        if indices:
            lines.append("**Indices**:")
            for idx in indices:
                unique = "UNIQUE " if idx["unique"] else ""
                cols = ", ".join(idx["columns"])
                lines.append(f"  - {unique}{idx['name']}: ({cols})")
            lines.append("")

        # Unique constraints
        unique_constraints = schema.get("unique_constraints")
        if unique_constraints:
            lines.append("**Unique Constraints**:")
            for constraint in unique_constraints:
                name = constraint.get("name") or "unnamed"
                cols = ", ".join(constraint["columns"])
                lines.append(f"  - {name}: ({cols})")
            lines.append("")

        # Check constraints
        check_constraints = schema.get("check_constraints")
        if check_constraints:
            lines.append("**Check Constraints**:")
            for constraint in check_constraints:
                name = constraint.get("name") or "unnamed"
                cols = ", ".join(constraint["columns"])
                lines.append(f"  - {name}: ({cols})")
            lines.append("")

        return "\n".join(lines)

    def _render_column(self, column: ColumnInfo) -> str:
        """
        Render a single column.

        Args:
            column: ColumnInfo with required fields (name, type, nullable) and optional default.
                   All required fields must be present and non-None.

        Returns:
            Formatted markdown string for the column
        """
        nullable = "NULL" if column["nullable"] else "NOT NULL"
        default = f" DEFAULT {column['default']}" if column.get("default") else ""
        return f"  - **{column['name']}**: `{column['type']}` {nullable}{default}"
