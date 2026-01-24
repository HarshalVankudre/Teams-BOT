"""
PostgreSQL Service for Hybrid RAG
Handles structured queries to the SEMA Matrix equipment database (sema_matrix).
"""
import os
import re
import time
from typing import List, Dict, Any, Optional, Sequence, Mapping, Union, Tuple
from dataclasses import dataclass

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from psycopg2.pool import ThreadedConnectionPool
    POSTGRES_AVAILABLE = True
    POOLING_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    POOLING_AVAILABLE = False
    print("[WARNING] psycopg2 not installed. PostgreSQL queries disabled.")

from .schema import DATABASE_SCHEMA


@dataclass
class PostgresConfig:
    """PostgreSQL configuration from environment"""
    host: str
    port: str
    database: str
    user: str
    password: str
    schema: str
    equipment_table: str
    sslmode: str = ""

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        return cls(
            host=os.getenv("POSTGRES_HOST", ""),
            port=os.getenv("POSTGRES_PORT", ""),
            database=os.getenv("POSTGRES_DB", ""),
            user=os.getenv("POSTGRES_USER", ""),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            schema=os.getenv("POSTGRES_SCHEMA", ""),
            equipment_table=os.getenv("POSTGRES_EQUIPMENT_TABLE", ""),
            sslmode=os.getenv("POSTGRES_SSLMODE", ""),
        )

    def validate(self) -> Optional[str]:
        missing = []
        if not self.host:
            missing.append("POSTGRES_HOST")
        if not self.port:
            missing.append("POSTGRES_PORT")
        if not self.database:
            missing.append("POSTGRES_DB")
        if not self.user:
            missing.append("POSTGRES_USER")
        if not self.password:
            missing.append("POSTGRES_PASSWORD")
        if not self.schema:
            missing.append("POSTGRES_SCHEMA")
        if not self.equipment_table:
            missing.append("POSTGRES_EQUIPMENT_TABLE")

        if missing:
            return f"Missing required Postgres env vars: {', '.join(missing)}"

        if not str(self.port).isdigit():
            return "Invalid POSTGRES_PORT (must be numeric)"

        if not _IDENTIFIER_RE.fullmatch(self.schema):
            return "Invalid POSTGRES_SCHEMA (must be a simple identifier)"
        if not _IDENTIFIER_RE.fullmatch(self.equipment_table):
            return "Invalid POSTGRES_EQUIPMENT_TABLE (must be a simple identifier)"

        return None

    def equipment_table_fqn(self) -> str:
        """Fully-qualified equipment table name (validated identifiers)."""
        return f"{self.schema}.{self.equipment_table}"

    def to_dict(self) -> Dict[str, str]:
        result = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }
        if self.sslmode:
            result["sslmode"] = self.sslmode
        return result


class PostgresService:
    """
    PostgreSQL service for structured equipment queries.
    Executes SQL queries against the SEMA Matrix schema (sema_matrix).
    """

    # Schema information imported from centralized schema.py
    SCHEMA_INFO = DATABASE_SCHEMA

    _READONLY_START_RE = re.compile(r"^(SELECT|WITH)\b", re.IGNORECASE)
    _DANGEROUS_KEYWORDS_RE = re.compile(
        r"\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _strip_leading_sql_comments(sql: str) -> str:
        """Remove leading SQL comments to make start-token validation reliable."""
        s = sql.lstrip()
        while True:
            if s.startswith("--"):
                newline_index = s.find("\n")
                if newline_index == -1:
                    return ""
                s = s[newline_index + 1 :].lstrip()
                continue
            if s.startswith("/*"):
                end_index = s.find("*/")
                if end_index == -1:
                    return ""
                s = s[end_index + 2 :].lstrip()
                continue
            return s

    @classmethod
    def prepare_readonly_sql(
        cls,
        sql: str,
        *,
        default_limit: int = 10000,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Validate and normalize readonly SQL (SELECT/CTE-only).

        Returns:
            (prepared_sql, error). If error is not None, prepared_sql will be None.
        """
        if not sql or not sql.strip():
            return None, "Empty SQL"

        sql = sql.strip()

        # Block multiple statements (allow a single trailing ';').
        if ";" in sql.rstrip(";"):
            return None, "Multiple statements are not allowed"

        # Normalize trailing semicolons.
        sql = sql.rstrip(";").strip()

        # Validate start token (ignore leading comments and optional parentheses).
        sql_for_checks = cls._strip_leading_sql_comments(sql)
        sql_start = sql_for_checks.lstrip().lstrip("(").lstrip()
        if not cls._READONLY_START_RE.match(sql_start):
            return None, "Only SELECT queries are allowed"

        # Block dangerous keywords (word-boundary match to avoid false positives like updated_at).
        match = cls._DANGEROUS_KEYWORDS_RE.search(sql_for_checks)
        if match:
            return None, f"Dangerous keyword '{match.group(1).upper()}' not allowed"

        # Basic truncation/format sanity check.
        if sql.count("(") != sql.count(")"):
            return None, "Malformed SQL: unbalanced parentheses"

        # Add a safety LIMIT if none specified (unless UNION is used).
        sql_upper = sql_for_checks.upper()
        is_union_query = "UNION" in sql_upper
        if "LIMIT" not in sql_upper and not is_union_query:
            sql = f"{sql} LIMIT {default_limit}"

        # Auto-convert LIKE to ILIKE for case-insensitive matching (PostgreSQL best practice)
        # This handles common LLM mistake of using case-sensitive LIKE
        import re
        sql = re.sub(r'\bLIKE\b', 'ILIKE', sql, flags=re.IGNORECASE)

        return sql, None

    def __init__(self, config: Optional[PostgresConfig] = None):
        self.config = config or PostgresConfig.from_env()
        config_error = self.config.validate()
        self.config_error = config_error
        self.equipment_table = self.config.equipment_table_fqn() if not config_error else None
        self.available = POSTGRES_AVAILABLE and (config_error is None)
        self.availability_error: Optional[str] = None
        self._column_cache: Optional[Dict[str, str]] = None
        self._pool: Optional[ThreadedConnectionPool] = None

        if self.available:
            # Initialize connection pool
            try:
                if POOLING_AVAILABLE:
                    self._pool = ThreadedConnectionPool(
                        minconn=1,
                        maxconn=5,
                        **self.config.to_dict()
                    )
                    # Test connection from pool
                    conn = self._pool.getconn()
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {self.equipment_table}")
                    count = cursor.fetchone()[0]
                    cursor.close()
                    self._pool.putconn(conn)
                    print(
                        f"[PostgreSQL] Connection pool initialized (1-5 connections), "
                        f"{count} equipment records ({self.equipment_table})"
                    )
                else:
                    # Fallback to per-request connections
                    conn = psycopg2.connect(**self.config.to_dict())
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {self.equipment_table}")
                    count = cursor.fetchone()[0]
                    cursor.close()
                    conn.close()
                    print(
                        f"[PostgreSQL] Connected (no pooling), "
                        f"{count} equipment records ({self.equipment_table})"
                    )
            except Exception as e:
                print(f"[PostgreSQL] Connection failed: {e}")
                self.availability_error = f"Connection failed: {e}"
                self.available = False
        else:
            reason = config_error or "psycopg2 not installed"
            print(f"[PostgreSQL] Service not available ({reason})")
            self.availability_error = reason

    def _get_connection(self):
        """Get database connection from pool or create new one."""
        if self._pool:
            return self._pool.getconn()
        return psycopg2.connect(**self.config.to_dict())

    def _release_connection(self, conn):
        """Release connection back to pool or close it."""
        if self._pool:
            self._pool.putconn(conn)
        else:
            conn.close()

    def close_pool(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            print("[PostgreSQL] Connection pool closed")

    def execute_query(
        self,
        sql: str,
        params: Optional[Union[Sequence[Any], Mapping[str, Any]]] = None,
        *,
        raise_on_error: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results as list of dicts.

        Args:
            sql: SQL query to execute
            params: Optional query parameters (recommended for non-LLM queries)
            raise_on_error: If True, raise exceptions instead of returning empty list

        Returns:
            List of result dictionaries
        """
        if not self.available:
            if raise_on_error:
                raise RuntimeError(self.availability_error or "PostgreSQL unavailable")
            return []

        query_start = time.time()

        conn_start = time.time()
        conn = self._get_connection()
        conn_ms = (time.time() - conn_start) * 1000

        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            exec_start = time.time()
            cursor.execute(sql, params)
            exec_ms = (time.time() - exec_start) * 1000

            fetch_start = time.time()
            results = [self._convert_row(dict(row)) for row in cursor.fetchall()]
            fetch_ms = (time.time() - fetch_start) * 1000

            total_ms = (time.time() - query_start) * 1000
            print(f"⏱️  [postgres:query] {total_ms:.0f}ms (conn={conn_ms:.0f}ms, exec={exec_ms:.0f}ms, fetch={fetch_ms:.0f}ms, rows={len(results)})")

            return results
        except Exception as e:
            print(f"[PostgreSQL] Query error: {e}")
            print(f"[PostgreSQL] SQL: {sql[:200]}...")
            if raise_on_error:
                raise
            return []
        finally:
            cursor.close()
            self._release_connection(conn)

    def get_column_info(self, refresh: bool = False) -> Dict[str, str]:
        """Fetch column metadata for the equipment table."""
        if not self.available:
            return {}
        if self._column_cache is not None and not refresh:
            return self._column_cache

        sql = """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """
        try:
            rows = self.execute_query(sql, (self.config.schema, self.config.equipment_table))
        except Exception as e:
            print(f"[PostgreSQL] Column lookup failed: {e}")
            return {}

        self._column_cache = {
            row.get("column_name"): row.get("data_type")
            for row in rows
            if row.get("column_name")
        }
        return self._column_cache

    def _convert_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Decimal and other non-JSON-serializable types to native Python types"""
        from decimal import Decimal
        for key, value in row.items():
            if isinstance(value, Decimal):
                row[key] = float(value)
        return row

    def get_equipment_count(
        self,
        category: Optional[str] = None,
        manufacturer: Optional[str] = None
    ) -> int:
        """
        Get count of equipment matching criteria.
        """
        sql = f"""
            SELECT COUNT(*) AS count
            FROM {self.equipment_table}
            WHERE 1=1
        """
        params: List[Any] = []

        if category:
            sql += " AND (geraetegruppe_name ILIKE %s OR geraetegruppe_code ILIKE %s)"
            params.append(f"%{category}%")
            params.append(f"%{category}%")
        if manufacturer:
            sql += " AND (hersteller_name ILIKE %s OR hersteller_code ILIKE %s)"
            params.append(f"%{manufacturer}%")
            params.append(f"%{manufacturer}%")

        results = self.execute_query(sql, params)
        return results[0]["count"] if results else 0

    def get_equipment_by_category(self) -> List[Dict[str, Any]]:
        """Get equipment counts by geraetegruppe_name (legacy method name)."""
        sql = f"""
            SELECT geraetegruppe_name AS equipment_group, COUNT(*) AS count
            FROM {self.equipment_table}
            WHERE geraetegruppe_name IS NOT NULL
            GROUP BY geraetegruppe_name
            ORDER BY count DESC
        """
        return self.execute_query(sql)

    def get_equipment_by_manufacturer(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get equipment counts by manufacturer"""
        sql = f"""
            SELECT hersteller_name AS manufacturer, COUNT(*) AS count
            FROM {self.equipment_table}
            WHERE hersteller_name IS NOT NULL
            GROUP BY hersteller_name
            ORDER BY count DESC
            LIMIT %s
        """
        return self.execute_query(sql, (limit,))

    def search_equipment(
        self,
        category: Optional[str] = None,
        manufacturer: Optional[str] = None,
        features: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search equipment with filters.

        Args:
            category: Filter by equipment_group (legacy name)
            manufacturer: Filter by manufacturer
            features: Property filters by name (e.g., {"Klimaanlage": True} or {"Gewicht": "2000"})
            limit: Max results

        Returns:
            List of matching equipment
        """
        sql = f"""
            SELECT
                id,
                bezeichnung,
                hersteller_name,
                hersteller_code,
                geraetegruppe_name,
                geraetegruppe_code,
                verwendung_code,
                verwendung_name,
                seriennummer,
                inventarnummer,
                nuclos_state,
                nuclos_process,
                prop_gewicht,
                prop_motor_leistung,
                prop_klimaanlage
            FROM {self.equipment_table}
            WHERE 1=1
        """
        params: List[Any] = []

        if category:
            sql += " AND (geraetegruppe_name ILIKE %s OR geraetegruppe_code ILIKE %s)"
            params.append(f"%{category}%")
            params.append(f"%{category}%")
        if manufacturer:
            sql += " AND (hersteller_name ILIKE %s OR hersteller_code ILIKE %s)"
            params.append(f"%{manufacturer}%")
            params.append(f"%{manufacturer}%")

        if features:
            for feature, desired in features.items():
                col_name = feature if str(feature).startswith("prop_") else f"prop_{feature}"
                if not re.fullmatch(r"[a-zA-Z0-9_]+", col_name or ""):
                    continue
                if isinstance(desired, bool) or desired is None:
                    sql += f" AND {col_name} = %s"
                    params.append(desired)
                else:
                    sql += f" AND CAST({col_name} AS TEXT) ILIKE %s"
                    params.append(f"%{desired}%")

        sql += " ORDER BY hersteller_name, bezeichnung LIMIT %s"
        params.append(limit)

        return self.execute_query(sql, params)

    def get_equipment_by_id(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Get single equipment row by id."""
        results = self.execute_query(
            f"SELECT * FROM {self.equipment_table} WHERE id = %s",
            (equipment_id,),
        )
        return results[0] if results else None

    def search_by_serial_or_inventory(
        self,
        serial_number: Optional[str] = None,
        inventory_number: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search by serial or inventory number."""
        clauses: List[str] = []
        params: List[Any] = []

        if serial_number:
            clauses.append("seriennummer ILIKE %s")
            params.append(f"%{serial_number}%")
        if inventory_number:
            clauses.append("inventarnummer ILIKE %s")
            params.append(f"%{inventory_number}%")

        if not clauses:
            return []

        sql = f"""
            SELECT
                id,
                bezeichnung,
                hersteller_name,
                hersteller_code,
                geraetegruppe_name,
                geraetegruppe_code,
                verwendung_code,
                verwendung_name,
                seriennummer,
                inventarnummer,
                nuclos_state,
                nuclos_process,
                prop_gewicht,
                prop_motor_leistung,
                prop_klimaanlage
            FROM {self.equipment_table}
            WHERE {" OR ".join(clauses)}
            LIMIT 10
        """
        return self.execute_query(sql, params)

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        stats = {
            "total_count": 0,
            "by_category": [],
            "by_manufacturer": [],
            "by_usage": []
        }

        # Total count
        total = self.execute_query(f"SELECT COUNT(*) as count FROM {self.equipment_table}")
        stats["total_count"] = total[0]["count"] if total else 0

        # By category
        stats["by_category"] = self.get_equipment_by_category()

        # By manufacturer (top 5)
        stats["by_manufacturer"] = self.get_equipment_by_manufacturer(5)

        # By usage
        usage = self.execute_query(
            f"""
            SELECT verwendung_code, verwendung_name, COUNT(*) AS count
            FROM {self.equipment_table}
            WHERE verwendung_code IS NOT NULL
            GROUP BY verwendung_code, verwendung_name
            ORDER BY count DESC
            """
        )
        stats["by_usage"] = usage

        return stats

    def search_by_use_case(self, use_case: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search equipment by use case using geraetegruppe_name and bezeichnung."""
        sql = f"""
            SELECT
                id,
                bezeichnung,
                hersteller_name,
                geraetegruppe_name,
                verwendung_code,
                seriennummer,
                inventarnummer
            FROM {self.equipment_table}
            WHERE geraetegruppe_name ILIKE %s
               OR bezeichnung ILIKE %s
            LIMIT %s
        """
        pattern = f"%{use_case}%"
        return self.execute_query(sql, (pattern, pattern, limit))

    def execute_dynamic_sql(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute dynamically generated SQL (from LLM).
        Includes safety checks.

        Args:
            sql: SQL query to execute

        Returns:
            Query results
        """
        prepared_sql, error = self.prepare_readonly_sql(sql, default_limit=10000)
        if error:
            print(f"[PostgreSQL] Blocked SQL: {error}")
            return []

        return self.execute_query(prepared_sql)
