from typing import Dict, List, Any
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from schema_search.types import (
    TableSchema,
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    ConstraintInfo,
)


class SchemaExtractor:
    def __init__(self, engine: Engine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config

    def extract(self) -> Dict[str, TableSchema]:
        inspector = inspect(self.engine)
        schemas: Dict[str, TableSchema] = {}

        for table_name in inspector.get_table_names():
            schemas[table_name] = self._extract_table(inspector, table_name)

        return schemas

    def _extract_table(self, inspector, table_name: str) -> TableSchema:
        pk_constraint = inspector.get_pk_constraint(table_name)

        schema: TableSchema = {
            "name": table_name,
            "columns": (
                self._extract_columns(inspector.get_columns(table_name))
                if self.config["schema"]["include_columns"]
                else None
            ),
            "primary_keys": pk_constraint["constrained_columns"],
            "foreign_keys": (
                self._extract_foreign_keys(inspector.get_foreign_keys(table_name))
                if self.config["schema"]["include_foreign_keys"]
                else None
            ),
            "indices": (
                self._extract_indices(inspector.get_indexes(table_name))
                if self.config["schema"]["include_indices"]
                else None
            ),
            "unique_constraints": (
                self._extract_constraints(inspector.get_unique_constraints(table_name))
                if self.config["schema"]["include_constraints"]
                else None
            ),
            "check_constraints": (
                self._extract_constraints(inspector.get_check_constraints(table_name))
                if self.config["schema"]["include_constraints"]
                else None
            ),
        }

        return schema

    def _extract_columns(self, columns: List[Dict[str, Any]]) -> List[ColumnInfo]:
        return [
            {
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col["nullable"],
                "default": str(col["default"]) if col["default"] else None,
            }
            for col in columns
        ]

    def _extract_foreign_keys(
        self, foreign_keys: List[Dict[str, Any]]
    ) -> List[ForeignKeyInfo]:
        return [
            {
                "constrained_columns": fk["constrained_columns"],
                "referred_table": fk["referred_table"],
                "referred_columns": fk["referred_columns"],
            }
            for fk in foreign_keys
        ]

    def _extract_indices(self, indices: List[Dict[str, Any]]) -> List[IndexInfo]:
        return [
            {
                "name": idx["name"],
                "columns": idx["column_names"],
                "unique": idx["unique"],
            }
            for idx in indices
        ]

    def _extract_constraints(
        self, constraints: List[Dict[str, Any]]
    ) -> List[ConstraintInfo]:
        return [
            {
                "name": constraint["name"],
                "columns": constraint["column_names"],
            }
            for constraint in constraints
        ]
