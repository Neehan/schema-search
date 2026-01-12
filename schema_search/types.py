from typing import TypedDict, List, Literal, Optional, Union
from dataclasses import dataclass, field


SearchType = Literal["semantic", "fuzzy", "bm25", "hybrid"]
OutputFormat = Literal["json", "markdown"]


class ColumnInfo(TypedDict):
    name: str
    type: str
    nullable: bool
    default: Optional[str]


class ForeignKeyInfo(TypedDict):
    constrained_columns: List[str]
    referred_table: str
    referred_columns: List[str]


class IndexInfo(TypedDict):
    name: str
    columns: List[str]
    unique: bool


class ConstraintInfo(TypedDict):
    name: Optional[str]
    columns: List[str]


class TableSchema(TypedDict):
    name: str
    primary_keys: List[str]
    columns: Optional[List[ColumnInfo]]
    foreign_keys: Optional[List[ForeignKeyInfo]]
    indices: Optional[List[IndexInfo]]
    unique_constraints: Optional[List[ConstraintInfo]]
    check_constraints: Optional[List[ConstraintInfo]]


class IndexResult(TypedDict):
    tables: int
    chunks: int
    latency_sec: float


class SearchResultItem(TypedDict):
    table: str
    score: float
    schema: TableSchema
    matched_chunks: List[str]
    related_tables: List[str]


@dataclass
class SearchResult:
    """Search result object with rendering capabilities."""

    results: List[SearchResultItem]
    latency_sec: float
    output_format: str = field(default="markdown")

    def __str__(self) -> str:
        """Render results using configured format."""
        from schema_search.renderers.factory import create_renderer
        renderer = create_renderer(self.output_format)
        return renderer.render(self)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return {
            "results": self.results,
            "latency_sec": self.latency_sec
        }
