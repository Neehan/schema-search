import logging
from typing import Dict, List, Any
from sqlalchemy.engine import Engine
from sqlalchemy import text

from schema_search.types import (
    TableSchema,
    ColumnInfo,
    ForeignKeyInfo,
    IndexInfo,
    ConstraintInfo,
)

logger = logging.getLogger(__name__)


class DatabricksSchemaExtractor:
    def __init__(self, engine: Engine, config: Dict[str, Any]):
        self.engine = engine
        self.config = config
        self.catalog = engine.url.database

    def extract(self) -> Dict[str, TableSchema]:
        logger.info("Starting extraction...")
        tables = self._get_tables()
        logger.info(f"Found {len(tables)} tables")

        logger.debug("Fetching all columns...")
        all_columns = self._get_all_columns() if self.config["schema"]["include_columns"] else {}

        logger.debug("Fetching all primary keys...")
        all_primary_keys = self._get_all_primary_keys()

        logger.debug("Fetching all foreign keys...")
        all_foreign_keys = self._get_all_foreign_keys() if self.config["schema"]["include_foreign_keys"] else {}

        schemas: Dict[str, TableSchema] = {}
        for table_name, table_schema in tables:
            table_key = (table_schema, table_name)

            schemas[table_name] = {
                "name": table_name,
                "columns": all_columns.get(table_key),
                "primary_keys": all_primary_keys.get(table_key, []),
                "foreign_keys": all_foreign_keys.get(table_key),
                "indices": [] if self.config["schema"]["include_indices"] else None,
                "unique_constraints": [] if self.config["schema"]["include_constraints"] else None,
                "check_constraints": [] if self.config["schema"]["include_constraints"] else None,
            }

        return schemas

    def _get_tables(self) -> List[tuple]:
        url_catalog = self.engine.url.database
        query = text(f"""
            SELECT table_name, table_schema, table_type
            FROM system.information_schema.tables
            WHERE table_catalog = :catalog
            AND table_schema NOT IN ('information_schema', 'sys')
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"catalog": url_catalog})
            rows = [(row[0], row[1]) for row in result]
            logger.debug(f"Found {len(rows)} tables in catalog {url_catalog}")
            return rows

    def _get_all_columns(self) -> Dict[tuple, List[ColumnInfo]]:
        query = text(f"""
            SELECT
                table_schema,
                table_name,
                column_name,
                data_type,
                is_nullable,
                column_default,
                ordinal_position
            FROM {self.catalog}.information_schema.columns
            WHERE table_schema NOT IN ('information_schema', 'sys')
            ORDER BY table_schema, table_name, ordinal_position
        """)

        columns_by_table = {}
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                table_key = (row[0], row[1])
                if table_key not in columns_by_table:
                    columns_by_table[table_key] = []

                columns_by_table[table_key].append({
                    "name": row[2],
                    "type": row[3],
                    "nullable": row[4] == "YES",
                    "default": row[5],
                })

        return columns_by_table

    def _get_all_primary_keys(self) -> Dict[tuple, List[str]]:
        query = text(f"""
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name,
                kcu.ordinal_position
            FROM {self.catalog}.information_schema.table_constraints tc
            JOIN {self.catalog}.information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
                AND tc.table_name = kcu.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
            ORDER BY tc.table_schema, tc.table_name, kcu.ordinal_position
        """)

        pks_by_table = {}
        with self.engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                table_key = (row[0], row[1])
                if table_key not in pks_by_table:
                    pks_by_table[table_key] = []
                pks_by_table[table_key].append(row[2])

        return pks_by_table

    def _get_all_foreign_keys(self) -> Dict[tuple, List[ForeignKeyInfo]]:
        query = text(f"""
            SELECT
                tc.table_schema,
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name
            FROM {self.catalog}.information_schema.table_constraints tc
            JOIN {self.catalog}.information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN {self.catalog}.information_schema.referential_constraints rc
                ON tc.constraint_name = rc.constraint_name
                AND tc.table_schema = rc.constraint_schema
            JOIN {self.catalog}.information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name = ccu.constraint_name
                AND rc.unique_constraint_schema = ccu.constraint_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
        """)

        fks_by_table = {}
        with self.engine.connect() as conn:
            result = conn.execute(query)

            for row in result:
                table_key = (row[0], row[1])
                col_name = row[2]
                ref_table = row[3]
                ref_col = row[4]

                if table_key not in fks_by_table:
                    fks_by_table[table_key] = {}

                if ref_table not in fks_by_table[table_key]:
                    fks_by_table[table_key][ref_table] = {
                        "constrained_columns": [],
                        "referred_table": ref_table,
                        "referred_columns": [],
                    }

                fks_by_table[table_key][ref_table]["constrained_columns"].append(col_name)
                fks_by_table[table_key][ref_table]["referred_columns"].append(ref_col)

        return {k: list(v.values()) for k, v in fks_by_table.items()}
