"""
PostgreSQL Service for Hybrid RAG
Handles structured queries to the SEMA equipment database on GCP Cloud SQL.
"""
import os
import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[WARNING] psycopg2 not installed. PostgreSQL queries disabled.")

from .schema import DATABASE_SCHEMA


@dataclass
class PostgresConfig:
    """PostgreSQL configuration from environment"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: str = os.getenv("POSTGRES_PORT", "5432")
    database: str = os.getenv("POSTGRES_DB", "sema")
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    def to_dict(self) -> Dict[str, str]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }


class PostgresService:
    """
    PostgreSQL service for structured equipment queries.
    Executes SQL queries against the SEMA database.
    """

    # Schema information imported from centralized schema.py
    SCHEMA_INFO = DATABASE_SCHEMA

    def __init__(self, config: Optional[PostgresConfig] = None):
        self.config = config or PostgresConfig()
        self.available = POSTGRES_AVAILABLE and bool(self.config.password)

        if self.available:
            # Test connection
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM geraete")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                print(f"[PostgreSQL] Connected to {self.config.host}, {count} equipment records")
            except Exception as e:
                print(f"[PostgreSQL] Connection failed: {e}")
                self.available = False
        else:
            print("[PostgreSQL] Service not available (missing credentials or psycopg2)")

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.config.to_dict())

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute SQL query and return results as list of dicts.

        Args:
            sql: SQL query to execute

        Returns:
            List of result dictionaries
        """
        if not self.available:
            return []

        conn = self._get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute(sql)
            results = [self._convert_row(dict(row)) for row in cursor.fetchall()]
            return results
        except Exception as e:
            print(f"[PostgreSQL] Query error: {e}")
            print(f"[PostgreSQL] SQL: {sql[:200]}...")
            return []
        finally:
            cursor.close()
            conn.close()

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
        """Get count of equipment matching criteria"""
        sql = "SELECT COUNT(*) as count FROM geraete WHERE 1=1"

        if category:
            sql += f" AND kategorie ILIKE '%{category}%'"
        if manufacturer:
            sql += f" AND hersteller ILIKE '%{manufacturer}%'"

        results = self.execute_query(sql)
        return results[0]["count"] if results else 0

    def get_equipment_by_category(self) -> List[Dict[str, Any]]:
        """Get equipment counts by category"""
        sql = """
            SELECT kategorie, COUNT(*) as count
            FROM geraete
            WHERE kategorie IS NOT NULL
            GROUP BY kategorie
            ORDER BY count DESC
        """
        return self.execute_query(sql)

    def get_equipment_by_manufacturer(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get equipment counts by manufacturer"""
        sql = f"""
            SELECT hersteller, COUNT(*) as count
            FROM geraete
            WHERE hersteller IS NOT NULL
            GROUP BY hersteller
            ORDER BY count DESC
            LIMIT {limit}
        """
        return self.execute_query(sql)

    def search_equipment(
        self,
        category: Optional[str] = None,
        manufacturer: Optional[str] = None,
        features: Optional[Dict[str, bool]] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search equipment with filters.

        Args:
            category: Filter by category
            manufacturer: Filter by manufacturer
            features: Boolean features to filter (e.g., {"klimaanlage": True})
            limit: Max results

        Returns:
            List of matching equipment
        """
        sql = """
            SELECT
                id, bezeichnung, hersteller, kategorie, geraetegruppe,
                seriennummer, inventarnummer, verwendung,
                eigenschaften_json->>'gewicht_kg' as gewicht_kg,
                eigenschaften_json->>'motor_leistung_kw' as motor_leistung_kw,
                eigenschaften_json->>'arbeitsbreite_mm' as arbeitsbreite_mm,
                eigenschaften_json->>'klimaanlage' as klimaanlage,
                eigenschaften_json->>'motor_hersteller' as motor_hersteller,
                eigenschaften_json->>'abgasstufe_eu' as abgasstufe_eu
            FROM geraete
            WHERE 1=1
        """

        if category:
            sql += f" AND (kategorie ILIKE '%{category}%' OR geraetegruppe ILIKE '%{category}%')"
        if manufacturer:
            sql += f" AND hersteller ILIKE '%{manufacturer}%'"
        if features:
            for feature, value in features.items():
                sql += f" AND (eigenschaften_json->>'{feature}')::boolean = {str(value).lower()}"

        sql += f" ORDER BY hersteller, bezeichnung LIMIT {limit}"

        return self.execute_query(sql)

    def get_equipment_by_id(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """Get single equipment by ID with all properties"""
        sql = f"""
            SELECT
                id, bezeichnung, hersteller, kategorie, geraetegruppe,
                seriennummer, inventarnummer, verwendung,
                inhalt, titel, eigenschaften_json
            FROM geraete
            WHERE id = '{equipment_id}'
        """
        results = self.execute_query(sql)
        return results[0] if results else None

    def search_by_serial_or_inventory(
        self,
        serial_number: Optional[str] = None,
        inventory_number: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search by serial or inventory number"""
        conditions = []
        if serial_number:
            conditions.append(f"seriennummer ILIKE '%{serial_number}%'")
        if inventory_number:
            conditions.append(f"inventarnummer ILIKE '%{inventory_number}%'")

        if not conditions:
            return []

        sql = f"""
            SELECT
                id, bezeichnung, hersteller, kategorie, geraetegruppe,
                seriennummer, inventarnummer, verwendung,
                eigenschaften_json
            FROM geraete
            WHERE {' OR '.join(conditions)}
            LIMIT 10
        """
        return self.execute_query(sql)

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics"""
        stats = {
            "total_count": 0,
            "by_category": [],
            "by_manufacturer": [],
            "by_usage": []
        }

        # Total count
        total = self.execute_query("SELECT COUNT(*) as count FROM geraete")
        stats["total_count"] = total[0]["count"] if total else 0

        # By category
        stats["by_category"] = self.get_equipment_by_category()

        # By manufacturer (top 5)
        stats["by_manufacturer"] = self.get_equipment_by_manufacturer(5)

        # By usage
        usage = self.execute_query("""
            SELECT verwendung, COUNT(*) as count
            FROM geraete
            WHERE verwendung IS NOT NULL
            GROUP BY verwendung
            ORDER BY count DESC
        """)
        stats["by_usage"] = usage

        return stats

    def search_by_use_case(self, use_case: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search equipment by use case (einsatzgebiete array)"""
        sql = f"""
            SELECT
                id, bezeichnung, hersteller, kategorie, geraetegruppe,
                einsatzgebiete, inhalt
            FROM geraete
            WHERE '{use_case}' = ANY(einsatzgebiete)
               OR inhalt ILIKE '%{use_case}%'
            LIMIT {limit}
        """
        return self.execute_query(sql)

    def execute_dynamic_sql(self, sql: str) -> List[Dict[str, Any]]:
        """
        Execute dynamically generated SQL (from LLM).
        Includes safety checks.

        Args:
            sql: SQL query to execute

        Returns:
            Query results
        """
        # Clean up SQL
        sql = sql.strip()
        sql_upper = sql.upper()

        # Block dangerous operations
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE']
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                print(f"[PostgreSQL] Blocked dangerous SQL: {keyword}")
                return []

        # Must be a SELECT query (including WITH/CTE, UNION variants)
        valid_starts = ('SELECT', '(SELECT', 'WITH')
        if not any(sql_upper.startswith(start) for start in valid_starts):
            print("[PostgreSQL] Only SELECT queries allowed")
            return []

        # Validate SQL has balanced parentheses (detect truncated SQL)
        if sql.count('(') != sql.count(')'):
            print("[PostgreSQL] Malformed SQL: unbalanced parentheses")
            return []

        # Add high safety LIMIT only if none specified
        # This prevents runaway queries while allowing full result sets
        sql = sql.rstrip(';')
        is_union_query = 'UNION' in sql_upper
        if 'LIMIT' not in sql_upper and not is_union_query:
            sql += ' LIMIT 10000'

        return self.execute_query(sql)


# Global instance
postgres_service = PostgresService()
