"""LangGraph tool definitions for the equipment assistant."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain_core.tools import tool

from rag.postgres import PostgresService
from rag.vector_store import PineconeStore

logger = logging.getLogger(__name__)

ALLOWED_OPERATORS = {">=", "<=", ">", "<", "="}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "numeric",
    "decimal",
    "real",
    "double precision",
}
MAX_RESULTS = 50
MAX_LIMIT = 20
DEFAULT_LIST_LIMIT = 10
SUMMARY_COLUMNS = (
    "id",
    "bezeichnung",
    "hersteller_name",
    "geraetegruppe_name",
    "verwendung_code",
    "seriennummer",
    "inventarnummer",
)


@dataclass(frozen=True)
class ColumnAliasRule:
    """Heuristic mapping from user intent to preferred schema columns."""

    keywords: Tuple[str, ...]
    preferred_columns: Tuple[str, ...]
    categories: Tuple[str, ...] = ()
    description: str = ""


COMMON_COLUMN_ALIASES = (
    ColumnAliasRule(
        keywords=("einbaubreite", "bohle", "verbreiterung"),
        preferred_columns=(
            "einbaubreite_max_m_num",
            "einbaubreite_grundbohle_m_num",
            "einbaubreite_verbreiterungen_m_num",
        ),
        categories=("fertiger",),
        description="Einbaubreite bei Fertigern",
    ),
    ColumnAliasRule(
        keywords=("arbeitsbreite",),
        preferred_columns=("arbeitsbreite_mm_num", "breite_mm_num"),
        description="Arbeitsbreite",
    ),
    ColumnAliasRule(
        keywords=("breite",),
        preferred_columns=("breite_mm_num",),
        description="Maschinenbreite",
    ),
    ColumnAliasRule(
        keywords=("hoehe",),
        preferred_columns=("hoehe_mm_num",),
        description="Maschinenhoehe",
    ),
    ColumnAliasRule(
        keywords=("grabtiefe",),
        preferred_columns=("grabtiefe_mm_num",),
        categories=("bagger",),
        description="Grabtiefe bei Baggern",
    ),
    ColumnAliasRule(
        keywords=("fraesbreite",),
        preferred_columns=("fraesbreite_mm_num",),
        categories=("fraese",),
        description="Fraesbreite bei Fraesen",
    ),
    ColumnAliasRule(
        keywords=("fraestiefe",),
        preferred_columns=("fraestiefe_mm_num",),
        categories=("fraese",),
        description="Fraestiefe bei Fraesen",
    ),
    ColumnAliasRule(
        keywords=("bohle", "bohlentyp"),
        preferred_columns=("prop_e2970_bohle_typ",),
        categories=("fertiger",),
        description="Bohle oder Bohlentyp bei Fertigern",
    ),
    ColumnAliasRule(
        keywords=("hgt", "schotter"),
        preferred_columns=("prop_e3070_einbau_von_hgt_schotter",),
        categories=("fertiger",),
        description="HGT-/Schottereinbau",
    ),
)

_postgres: Optional[PostgresService] = None
_pinecone: Optional[PineconeStore] = None


def set_shared_postgres(postgres: PostgresService) -> None:
    """Inject a shared PostgreSQL service instance."""
    global _postgres
    _postgres = postgres


def set_shared_pinecone(pinecone: PineconeStore) -> None:
    """Inject a shared Pinecone service instance."""
    global _pinecone
    _pinecone = pinecone


def _get_postgres() -> PostgresService:
    global _postgres
    if _postgres is None:
        _postgres = PostgresService()
    return _postgres


def _get_pinecone() -> PineconeStore:
    global _pinecone
    if _pinecone is None:
        _pinecone = PineconeStore()
    return _pinecone


def _column_catalog() -> Dict[str, str]:
    return _get_postgres().get_column_info()


def _validate_known_column(column_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not column_name:
        return None, "Column name is required"
    if not IDENTIFIER_RE.fullmatch(column_name):
        return None, f"Invalid column name: {column_name}"

    columns = _column_catalog()
    if columns and column_name not in columns:
        return None, (
            f"Unknown column '{column_name}'. "
            "Call list_filter_columns() first if you are unsure which field to use."
        )
    return column_name, None


def _is_numeric_column(column_name: str, column_type: Optional[str]) -> bool:
    return column_name.endswith("_num") or (column_type or "").lower() in NUMERIC_TYPES


def _clean_limit(limit: int, *, default: int = DEFAULT_LIST_LIMIT) -> int:
    return max(1, min(int(limit or default), MAX_LIMIT))


def _clean_text(value: Optional[str]) -> str:
    normalized = (value or "").lower()
    normalized = normalized.replace("ae", "a").replace("oe", "o").replace("ue", "u")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _nuclos_status_note(state: Optional[str]) -> str:
    if state == "Released":
        return "Released bedeutet nur Bestandsstatus im System, nicht gesicherte Live-Verfuegbarkeit."
    if state == "Locked":
        return "Locked bedeutet gesperrt bzw. nicht freigegeben im System."
    if state:
        return f"Nuclos-Status: {state}"
    return "Kein Nuclos-Status vorhanden."


def _build_base_filters(
    *,
    category: Optional[str] = None,
    manufacturer: Optional[str] = None,
    usage_type: Optional[str] = None,
    property_filters: Sequence[Tuple[Optional[str], Optional[str], Union[str, int, float, None]]] = (),
) -> Tuple[List[str], List[Any], Optional[str]]:
    clauses = ["1=1"]
    params: List[Any] = []

    if category:
        clauses.append("geraetegruppe_name ILIKE %s")
        params.append(f"%{category}%")

    if manufacturer:
        clauses.append("hersteller_name ILIKE %s")
        params.append(f"%{manufacturer}%")

    if usage_type:
        clauses.append("verwendung_code = %s")
        params.append(str(usage_type).upper())

    for column_name, operator, value in property_filters:
        clause, clause_params, error = _build_property_clause(column_name, operator, value)
        if error:
            return [], [], error
        if clause:
            clauses.append(clause)
            params.extend(clause_params)

    return clauses, params, None


def _match_alias_rules(intent: str, category: Optional[str], columns: Dict[str, str]) -> List[Dict[str, Any]]:
    normalized_intent = _clean_text(intent)
    normalized_category = _clean_text(category)
    matches: List[Dict[str, Any]] = []

    for rule in COMMON_COLUMN_ALIASES:
        if not any(_clean_text(keyword) in normalized_intent for keyword in rule.keywords):
            continue
        if rule.categories and not any(_clean_text(token) in normalized_category for token in rule.categories):
            continue

        available_columns = [
            column_name
            for column_name in rule.preferred_columns
            if column_name in columns
        ]
        if not available_columns:
            continue

        matches.append(
            {
                "reason": rule.description or "Alias match",
                "columns": [
                    {"name": column_name, "type": columns[column_name]}
                    for column_name in available_columns
                ],
            }
        )

    return matches


def _search_columns_by_intent(intent: str, columns: Dict[str, str], *, limit: int) -> List[Dict[str, Any]]:
    tokens = [token for token in _clean_text(intent).split() if len(token) >= 3]
    if not tokens:
        return []

    scored_matches: List[Tuple[int, int, str, str]] = []
    for column_name, data_type in columns.items():
        score = sum(1 for token in tokens if token in column_name.lower())
        if score:
            is_numeric = 1 if _is_numeric_column(column_name, data_type) else 0
            scored_matches.append((score, is_numeric, column_name, data_type))

    scored_matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [
        {"name": column_name, "type": data_type, "score": score}
        for score, _, column_name, data_type in scored_matches[:limit]
    ]


def _build_property_clause(
    column_name: Optional[str],
    operator: Optional[str],
    value: Union[str, int, float, None],
) -> Tuple[Optional[str], List[Any], Optional[str]]:
    if not column_name:
        return None, [], None
    if operator not in ALLOWED_OPERATORS:
        return None, [], f"Unsupported operator: {operator}"
    if value is None:
        return None, [], f"Missing value for column: {column_name}"

    validated_column, error = _validate_known_column(column_name)
    if error:
        return None, [], error

    column_type = _column_catalog().get(validated_column)
    if _is_numeric_column(validated_column, column_type):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None, [], f"Numeric column '{validated_column}' requires a number"
        return f'"{validated_column}" {operator} %s', [numeric_value], None

    if operator != "=":
        return None, [], f"Text column '{validated_column}' only supports '='"

    return f'CAST("{validated_column}" AS TEXT) ILIKE %s', [f"%{value}%"], None


def _run_select(
    sql: str,
    params: Optional[Sequence[Any]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    postgres = _get_postgres()
    prepared_sql, error = postgres.prepare_readonly_sql(sql, default_limit=MAX_RESULTS)
    if error:
        return [], error
    try:
        results = postgres.execute_query(prepared_sql, params=params, raise_on_error=True)
        return results, None
    except Exception as exc:  # pragma: no cover - depends on external services
        logger.error("SQL execution failed: %s", exc, exc_info=True)
        return [], str(exc)


@tool
def list_filter_columns(
    keyword: Optional[str] = None,
    only_numeric: bool = False,
    limit: int = 50,
) -> Dict[str, Any]:
    """List available database columns for structured filters."""
    columns = _column_catalog()
    if not columns:
        return {"error": "Column metadata unavailable"}

    normalized_keyword = (keyword or "").strip().lower()
    results = []
    for name, data_type in columns.items():
        if only_numeric and not _is_numeric_column(name, data_type):
            continue
        if normalized_keyword and normalized_keyword not in name.lower():
            continue
        results.append({"name": name, "type": data_type})

    results = results[: max(1, min(limit, 100))]
    return {
        "keyword": keyword or "",
        "only_numeric": only_numeric,
        "column_count": len(results),
        "columns": results,
    }


@tool
def suggest_filter_columns(
    intent: str,
    category: Optional[str] = None,
    limit: int = 8,
) -> Dict[str, Any]:
    """Suggest likely filter columns for a user intent before querying."""
    columns = _column_catalog()
    if not columns:
        return {"error": "Column metadata unavailable"}

    cleaned_limit = max(1, min(int(limit or 8), 20))
    alias_matches = _match_alias_rules(intent, category, columns)
    keyword_matches = _search_columns_by_intent(intent, columns, limit=cleaned_limit)

    suggested_names: List[str] = []
    seen_names = set()
    for match in alias_matches:
        for column in match["columns"]:
            if column["name"] not in seen_names:
                seen_names.add(column["name"])
                suggested_names.append(column["name"])
    for match in keyword_matches:
        if match["name"] not in seen_names:
            seen_names.add(match["name"])
            suggested_names.append(match["name"])

    return {
        "intent": intent,
        "category": category or "",
        "suggested_columns": suggested_names[:cleaned_limit],
        "alias_matches": alias_matches[:cleaned_limit],
        "keyword_matches": keyword_matches[:cleaned_limit],
    }


@tool
def execute_sql(sql: str, purpose: str) -> Dict[str, Any]:
    """Execute a validated read-only SQL query against the equipment database."""
    results, error = _run_select(sql)
    if error:
        return {"error": error, "sql": sql}

    result_ids = [row.get("id") for row in results if row.get("id") is not None]
    return {
        "purpose": purpose,
        "row_count": len(results),
        "results": results[:MAX_RESULTS],
        "result_ids": result_ids[:100],
    }


@tool
def count_equipment(
    category: Optional[str] = None,
    manufacturer: Optional[str] = None,
    property_column: Optional[str] = None,
    property_operator: Optional[str] = None,
    property_value: Union[str, int, float, None] = None,
    property_column_2: Optional[str] = None,
    property_operator_2: Optional[str] = None,
    property_value_2: Union[str, int, float, None] = None,
    usage_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Count equipment matching structured filters. Use this for 'Wie viele ...' questions."""
    postgres = _get_postgres()
    clauses, params, error = _build_base_filters(
        category=category,
        manufacturer=manufacturer,
        usage_type=usage_type,
        property_filters=(
            (property_column, property_operator, property_value),
            (property_column_2, property_operator_2, property_value_2),
        ),
    )
    if error:
        return {"error": error}

    sql = f"""
        SELECT COUNT(*) AS count
        FROM {postgres.equipment_table}
        WHERE {' AND '.join(clauses)}
    """
    results, error = _run_select(sql, params)
    if error:
        return {"error": error}

    return {
        "count": int((results[0] or {}).get("count", 0)) if results else 0,
        "category": category or "",
        "manufacturer": manufacturer or "",
        "usage_type": usage_type or "",
    }


@tool
def recommend_fertiger_for_width(
    target_width_m: float,
    usage_type: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Recommend pavers for a target installation width, including whether extensions are required."""
    postgres = _get_postgres()
    cleaned_limit = _clean_limit(limit, default=5)
    category_filter = category or "Fertiger"

    clauses = ["geraetegruppe_name ILIKE %s", "GREATEST(COALESCE(einbaubreite_max_m_num, 0), COALESCE(einbaubreite_verbreiterungen_m_num, 0), COALESCE(einbaubreite_grundbohle_m_num, 0)) >= %s"]
    params: List[Any] = [f"%{category_filter}%", float(target_width_m)]

    if usage_type:
        clauses.append("verwendung_code = %s")
        params.append(str(usage_type).upper())

    sql = f"""
        SELECT
            bezeichnung,
            hersteller_name,
            seriennummer,
            inventarnummer,
            geraetegruppe_name,
            verwendung_code,
            nuclos_state,
            einbaubreite_grundbohle_m_num,
            einbaubreite_max_m_num,
            einbaubreite_verbreiterungen_m_num,
            prop_e2970_bohle_typ
        FROM {postgres.equipment_table}
        WHERE {' AND '.join(clauses)}
        ORDER BY
            GREATEST(
                COALESCE(einbaubreite_max_m_num, 0),
                COALESCE(einbaubreite_verbreiterungen_m_num, 0),
                COALESCE(einbaubreite_grundbohle_m_num, 0)
            ) ASC NULLS LAST,
            einbaubreite_grundbohle_m_num DESC NULLS LAST,
            bezeichnung ASC
        LIMIT %s
    """
    params.append(cleaned_limit)

    results, error = _run_select(sql, params)
    if error:
        return {"error": error}

    matches: List[Dict[str, Any]] = []
    for row in results:
        base_width = row.get("einbaubreite_grundbohle_m_num")
        max_width = row.get("einbaubreite_max_m_num")
        extension_width = row.get("einbaubreite_verbreiterungen_m_num")
        supported_width = max(
            float(base_width or 0),
            float(max_width or 0),
            float(extension_width or 0),
        )
        requires_extensions = bool(base_width is not None and float(target_width_m) > float(base_width))
        matches.append(
            {
                "bezeichnung": row.get("bezeichnung"),
                "hersteller_name": row.get("hersteller_name"),
                "seriennummer": row.get("seriennummer"),
                "inventarnummer": row.get("inventarnummer"),
                "geraetegruppe_name": row.get("geraetegruppe_name"),
                "verwendung_code": row.get("verwendung_code"),
                "nuclos_state": row.get("nuclos_state"),
                "status_note": _nuclos_status_note(row.get("nuclos_state")),
                "einbaubreite_grundbohle_m_num": base_width,
                "einbaubreite_max_m_num": max_width,
                "einbaubreite_verbreiterungen_m_num": extension_width,
                "prop_e2970_bohle_typ": row.get("prop_e2970_bohle_typ"),
                "supported_width_m": supported_width,
                "requires_extensions": requires_extensions,
                "covers_on_base": bool(base_width is not None and float(target_width_m) <= float(base_width)),
                "width_margin_m": round(supported_width - float(target_width_m), 3),
            }
        )

    return {
        "target_width_m": float(target_width_m),
        "usage_type": (usage_type or "").upper(),
        "category": category_filter,
        "match_count": len(matches),
        "availability_note": "Nuclos-Status und Nutzungscode beschreiben nur Bestands-/Nutzungsstatus, nicht die echte Dispositionsverfuegbarkeit.",
        "recommended_machine": matches[0] if matches else None,
        "matches": matches,
    }


@tool
def find_hgt_fertiger(
    usage_type: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Find pavers flagged for HGT/Schotter installation in the database."""
    postgres = _get_postgres()
    cleaned_limit = _clean_limit(limit, default=10)
    clauses = [
        "geraetegruppe_name ILIKE %s",
        "prop_e3070_einbau_von_hgt_schotter ILIKE %s",
    ]
    params: List[Any] = ["%Fertiger%", "Ja%"]

    if usage_type:
        clauses.append("verwendung_code = %s")
        params.append(str(usage_type).upper())

    sql = f"""
        SELECT
            bezeichnung,
            hersteller_name,
            seriennummer,
            inventarnummer,
            geraetegruppe_name,
            verwendung_code,
            nuclos_state,
            prop_e3070_einbau_von_hgt_schotter,
            prop_e2970_bohle_typ,
            einbaubreite_grundbohle_m_num,
            einbaubreite_max_m_num,
            einbaubreite_verbreiterungen_m_num
        FROM {postgres.equipment_table}
        WHERE {' AND '.join(clauses)}
        ORDER BY hersteller_name NULLS LAST, bezeichnung NULLS LAST
        LIMIT %s
    """
    params.append(cleaned_limit)

    results, error = _run_select(sql, params)
    if error:
        return {"error": error}

    return {
        "usage_type": (usage_type or "").upper(),
        "match_count": len(results),
        "availability_note": "Nuclos-Status und Nutzungscode beschreiben nur Bestands-/Nutzungsstatus, nicht die echte Dispositionsverfuegbarkeit.",
        "matches": [
            {
                **row,
                "status_note": _nuclos_status_note(row.get("nuclos_state")),
            }
            for row in results
        ],
    }


@tool
def query_equipment(
    category: Optional[str] = None,
    manufacturer: Optional[str] = None,
    property_column: Optional[str] = None,
    property_operator: Optional[str] = None,
    property_value: Union[str, int, float, None] = None,
    property_column_2: Optional[str] = None,
    property_operator_2: Optional[str] = None,
    property_value_2: Union[str, int, float, None] = None,
    usage_type: Optional[str] = None,
    limit: int = DEFAULT_LIST_LIMIT,
) -> Dict[str, Any]:
    """List equipment rows with structured filters. Do not use for count-only questions."""
    postgres = _get_postgres()
    clauses, params, error = _build_base_filters(
        category=category,
        manufacturer=manufacturer,
        usage_type=usage_type,
        property_filters=(
            (property_column, property_operator, property_value),
            (property_column_2, property_operator_2, property_value_2),
        ),
    )
    if error:
        return {"error": error}

    cleaned_limit = _clean_limit(limit)
    select_columns = ", ".join(SUMMARY_COLUMNS)
    sql = f"""
        SELECT {select_columns}
        FROM {postgres.equipment_table}
        WHERE {' AND '.join(clauses)}
        ORDER BY hersteller_name NULLS LAST, bezeichnung NULLS LAST
        LIMIT %s
    """
    params.append(cleaned_limit)

    results, error = _run_select(sql, params)
    if error:
        return {"error": error}

    returned_count = len(results)
    return {
        "row_count": returned_count,
        "returned_count": returned_count,
        "results": results,
        "limit": cleaned_limit,
        "result_type": "sample_list",
    }


@tool
def lookup_equipment(search_term: str, include_fields: str = "all") -> Dict[str, Any]:
    """Look up equipment by model, serial number, or inventory number."""
    postgres = _get_postgres()
    fields = (
        "id, bezeichnung, hersteller_name, seriennummer, geraetegruppe_name"
        if include_fields == "basic"
        else """
            id, bezeichnung, hersteller_name, seriennummer, inventarnummer,
            geraetegruppe_name, verwendung_code, nuclos_state, properties_jsonb
        """
    )
    sql = f"""
        SELECT {fields}
        FROM {postgres.equipment_table}
        WHERE bezeichnung ILIKE %s
           OR seriennummer ILIKE %s
           OR inventarnummer ILIKE %s
        LIMIT 5
    """
    pattern = f"%{search_term}%"
    results, error = _run_select(sql, [pattern, pattern, pattern])
    if error:
        return {"error": error, "search_term": search_term}

    return {
        "search_term": search_term,
        "found": len(results),
        "machines": [
            {
                **row,
                "status_note": _nuclos_status_note(row.get("nuclos_state")),
            }
            for row in results
        ],
    }


@tool
def get_equipment_details(
    equipment_id: Union[int, str, None] = None,
    serial_number: Optional[str] = None,
    property_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Get detailed fields for a single machine."""
    postgres = _get_postgres()

    if equipment_id is not None:
        try:
            equipment_id = int(equipment_id)
        except (TypeError, ValueError):
            serial_number = str(equipment_id)
            equipment_id = None

    if equipment_id is None and not serial_number:
        return {"error": "Provide either equipment_id or serial_number"}

    if equipment_id is not None:
        sql = f"""
            SELECT id, bezeichnung, hersteller_name, seriennummer, inventarnummer,
                   geraetegruppe_name, verwendung_code, nuclos_state, properties_jsonb
            FROM {postgres.equipment_table}
            WHERE id = %s
            LIMIT 1
        """
        params: Sequence[Any] = [equipment_id]
    else:
        pattern = f"%{serial_number}%"
        sql = f"""
            SELECT id, bezeichnung, hersteller_name, seriennummer, inventarnummer,
                   geraetegruppe_name, verwendung_code, nuclos_state, properties_jsonb
            FROM {postgres.equipment_table}
            WHERE seriennummer ILIKE %s
               OR inventarnummer ILIKE %s
               OR bezeichnung ILIKE %s
            LIMIT 1
        """
        params = [pattern, pattern, pattern]

    results, error = _run_select(sql, params)
    if error:
        return {"error": error}
    if not results:
        return {"error": "No equipment found"}

    machine = results[0]
    properties = {
        key: value
        for key, value in (machine.get("properties_jsonb") or {}).items()
        if value not in (None, "")
    }

    if property_filter:
        needle = property_filter.lower()
        properties = {
            key: value
            for key, value in properties.items()
            if needle in key.lower() or needle in str(value).lower()
        }

    return {
        "id": machine.get("id"),
        "bezeichnung": machine.get("bezeichnung"),
        "hersteller": machine.get("hersteller_name"),
        "seriennummer": machine.get("seriennummer"),
        "inventarnummer": machine.get("inventarnummer"),
        "geraetegruppe": machine.get("geraetegruppe_name"),
        "verwendung": machine.get("verwendung_code"),
        "nuclos_state": machine.get("nuclos_state"),
        "status_note": _nuclos_status_note(machine.get("nuclos_state")),
        "property_count": len(properties),
        "properties": properties,
    }


@tool
async def search_documents(query: str, top_k: int = 10) -> Dict[str, Any]:
    """Search technical documents and manuals in Pinecone."""
    pinecone = _get_pinecone()
    cleaned_top_k = max(1, min(int(top_k or 10), 20))
    try:
        results = await pinecone.search(query, top_k=cleaned_top_k)
    except Exception as exc:  # pragma: no cover - depends on external services
        logger.error("Document search failed: %s", exc, exc_info=True)
        return {"error": str(exc), "query": query}

    matches = [
        {
            "title": row.get("metadata", {}).get("title", "Untitled"),
            "content": row.get("metadata", {}).get("content", "")[:500],
            "source": row.get("metadata", {}).get("source_file", "unknown"),
            "score": row.get("score", 0),
        }
        for row in results
    ]
    return {"query": query, "match_count": len(matches), "matches": matches}


@tool
def explore_column(column_name: str) -> Dict[str, Any]:
    """Show distinct values from a known database column."""
    validated_column, error = _validate_known_column(column_name)
    if error:
        return {"error": error}

    postgres = _get_postgres()
    sql = f"""
        SELECT DISTINCT "{validated_column}"
        FROM {postgres.equipment_table}
        WHERE "{validated_column}" IS NOT NULL
        ORDER BY "{validated_column}"
        LIMIT 50
    """
    results, error = _run_select(sql)
    if error:
        return {"error": error, "column": validated_column}

    values = [row.get(validated_column) for row in results if row.get(validated_column) is not None]
    return {
        "column": validated_column,
        "distinct_count": len(values),
        "values": values,
    }


def get_langgraph_tools() -> List[Any]:
    """Return the LangGraph tool list in the preferred order."""
    return [
        query_equipment,
        count_equipment,
        recommend_fertiger_for_width,
        find_hgt_fertiger,
        lookup_equipment,
        get_equipment_details,
        suggest_filter_columns,
        list_filter_columns,
        explore_column,
        execute_sql,
        search_documents,
    ]


__all__ = [
    "count_equipment",
    "execute_sql",
    "explore_column",
    "find_hgt_fertiger",
    "get_equipment_details",
    "get_langgraph_tools",
    "list_filter_columns",
    "lookup_equipment",
    "query_equipment",
    "recommend_fertiger_for_width",
    "search_documents",
    "set_shared_pinecone",
    "set_shared_postgres",
    "suggest_filter_columns",
]
