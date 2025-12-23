"""
SQL intent extraction and validation guardrails.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except Exception:
    SQLGLOT_AVAILABLE = False


_WHITESPACE_RE = re.compile(r"\s+")
_TOP_N_RE = re.compile(r"\btop\s+(\d+)\b")
_LIMIT_N_RE = re.compile(r"\b(?:zeige|liste|list|show|gib|give)\b.*?\b(\d+)\b")
_FOLLOWUP_RE = re.compile(r"\b(davon|diese|diesen|diese[nr]?|welche\s+davon|die\s+alle)\b")
_COUNT_RE = re.compile(r"\b(wie\s+viele|anzahl|count)\b")
_GROUP_BY_RE = re.compile(
    r"\b(pro|nach)\s+(hersteller|manufacturer|gerategruppe|geraetegruppe|gruppe|verwendung)\b"
)
_TIME_RANGE_RE = re.compile(
    r"\b(letzte[nr]?|letzten|seit|zwischen|in\s+den\s+letzten|last\s+\d+|last\s+month|last\s+year|\d{4})\b"
)
_DOC_RE = re.compile(r"\b(handbuch|anleitung|dokument|manual|pdf|richtlinie|policy)\b")
_STRUCTURED_RE = re.compile(
    r"\b(seriennummer|inventarnummer|hersteller|maschine|maschinen|"
    r"gerat|gerate|geraet|geraete|"
    r"miet|vermiet|verkauf|klimaanlage|prop_|nuclos)\b"
)
_RENTAL_RE = re.compile(r"\b(miet|vermiet|rental|rent)")
_SALES_RE = re.compile(r"\b(verkauf|vk|sale|buy|kauf)")
_AVAIL_RE = re.compile(r"\b(verfugbar|available|released|frei)")
_NULL_RE = re.compile(r"\b(ohne|keine|null)\b")
_AC_RE = re.compile(r"\b(klimaanlage|klima)\b")
_WEIGHT_RE = re.compile(r"\b(gewicht|schwer)\b")
_HEAVIEST_RE = re.compile(r"\b(schwerst|schwersten|heaviest)\b")
_POWER_RE = re.compile(r"\b(leistung|motorleistung|kw)\b")
_WIDTH_RE = re.compile(r"\b(einbaubreite|arbeitsbreite)\b")


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _normalize_sql(sql: str) -> str:
    return _WHITESPACE_RE.sub(" ", (sql or "").strip().lower())


@dataclass
class SQLConstraint:
    name: str
    description: str
    patterns: Sequence[re.Pattern]

    def is_satisfied(self, normalized_sql: str) -> bool:
        return any(pattern.search(normalized_sql) for pattern in self.patterns)


@dataclass
class SQLIntent:
    query: str
    requires_sql: bool
    prefers_sql: bool
    prefers_documents: bool
    request_type: str = "unknown"
    requested_limit: Optional[int] = None
    required_constraints: List[SQLConstraint] = field(default_factory=list)
    followup_ids: List[Any] = field(default_factory=list)
    clarification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "requires_sql": self.requires_sql,
            "prefers_sql": self.prefers_sql,
            "prefers_documents": self.prefers_documents,
            "request_type": self.request_type,
            "requested_limit": self.requested_limit,
            "required_constraints": [c.name for c in self.required_constraints],
            "followup_ids": self.followup_ids,
            "clarification": self.clarification,
        }


@dataclass
class SQLValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    normalized_sql: str = ""
    referenced_tables: List[str] = field(default_factory=list)
    referenced_columns: List[str] = field(default_factory=list)
    limit_value: Optional[int] = None


class SQLGuard:
    def __init__(
        self,
        *,
        equipment_table: Optional[str],
        column_resolver: Optional[Callable[[], Dict[str, str]]] = None,
    ):
        self.equipment_table = equipment_table
        self._column_resolver = column_resolver
        self._cached_columns: Optional[Dict[str, str]] = None

    def _get_columns(self) -> Dict[str, str]:
        if self._cached_columns is not None:
            return self._cached_columns
        if not self._column_resolver:
            self._cached_columns = {}
            return self._cached_columns
        try:
            self._cached_columns = self._column_resolver() or {}
        except Exception:
            self._cached_columns = {}
        return self._cached_columns

    def _guess_date_columns(self) -> List[str]:
        columns = self._get_columns()
        candidates = []
        for name in columns.keys():
            lowered = name.lower()
            if any(token in lowered for token in ("date", "datum", "created", "updated")):
                candidates.append(name)
        return candidates

    def _intent_requires_sql(self, normalized_query: str) -> bool:
        if _DOC_RE.search(normalized_query) and not _STRUCTURED_RE.search(normalized_query):
            return False
        if _STRUCTURED_RE.search(normalized_query):
            return True
        return bool(_COUNT_RE.search(normalized_query) or _LIMIT_N_RE.search(normalized_query))

    def _intent_prefers_documents(self, normalized_query: str) -> bool:
        return bool(_DOC_RE.search(normalized_query))

    def _intent_prefers_sql(self, normalized_query: str) -> bool:
        return bool(_STRUCTURED_RE.search(normalized_query) or _COUNT_RE.search(normalized_query))

    def extract_intent(
        self,
        user_query: str,
        *,
        thread_state: Optional[Dict[str, Any]] = None,
        manufacturer_matches: Optional[List[Dict[str, str]]] = None,
    ) -> SQLIntent:
        normalized_query = _normalize_text(user_query or "")
        requires_sql = self._intent_requires_sql(normalized_query)
        prefers_sql = self._intent_prefers_sql(normalized_query)
        prefers_documents = self._intent_prefers_documents(normalized_query)
        intent = SQLIntent(
            query=user_query,
            requires_sql=requires_sql,
            prefers_sql=prefers_sql,
            prefers_documents=prefers_documents,
        )

        if _COUNT_RE.search(normalized_query):
            intent.request_type = "count"
            intent.required_constraints.append(
                SQLConstraint(
                    name="count",
                    description="Use COUNT(*) for count questions.",
                    patterns=[re.compile(r"\bcount\s*\(", re.IGNORECASE)],
                )
            )

        top_match = _TOP_N_RE.search(normalized_query) or _LIMIT_N_RE.search(normalized_query)
        if top_match:
            try:
                intent.requested_limit = int(top_match.group(1))
            except (ValueError, TypeError):
                intent.requested_limit = None
            if intent.requested_limit:
                intent.required_constraints.append(
                    SQLConstraint(
                        name="limit",
                        description=f"Limit results to {intent.requested_limit}.",
                        patterns=[
                            re.compile(rf"\blimit\s+{intent.requested_limit}\b", re.IGNORECASE),
                            re.compile(rf"\bfetch\s+first\s+{intent.requested_limit}\b", re.IGNORECASE),
                        ],
                    )
                )

        if _GROUP_BY_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="group_by",
                    description="Use GROUP BY when user asks for grouping.",
                    patterns=[re.compile(r"\bgroup\s+by\b", re.IGNORECASE)],
                )
            )

        if _RENTAL_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="usage_rental",
                    description="Filter to rental usage (MIET).",
                    patterns=[
                        re.compile(r"\bverwendung_code\s*=\s*'MIET'\b", re.IGNORECASE),
                        re.compile(r"\bverwendung_code\s+in\s*\([^)]*'MIET'[^)]*\)", re.IGNORECASE),
                        re.compile(r"\bverwendung_name\s+ilike\s+'%miet%'", re.IGNORECASE),
                    ],
                )
            )

        if _SALES_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="usage_sales",
                    description="Filter to sales usage (VK).",
                    patterns=[
                        re.compile(r"\bverwendung_code\s*=\s*'VK'\b", re.IGNORECASE),
                        re.compile(r"\bverwendung_code\s+in\s*\([^)]*'VK'[^)]*\)", re.IGNORECASE),
                        re.compile(r"\bverwendung_name\s+ilike\s+'%verkauf%'", re.IGNORECASE),
                    ],
                )
            )

        if _AVAIL_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="availability_released",
                    description="Filter to released/available machines.",
                    patterns=[
                        re.compile(r"\bnuclos_state\s*=\s*'Released'\b", re.IGNORECASE),
                        re.compile(r"\bnuclos_state\s+ilike\s+'released'", re.IGNORECASE),
                    ],
                )
            )

        if _AC_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="prop_klimaanlage",
                    description="Filter by Klimaanlage when requested.",
                    patterns=[re.compile(r"\bprop_klimaanlage\b", re.IGNORECASE)],
                )
            )

        if _WEIGHT_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="prop_gewicht",
                    description="Use prop_gewicht for weight-related questions.",
                    patterns=[re.compile(r"\bprop_gewicht\b", re.IGNORECASE)],
                )
            )

        if _HEAVIEST_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="order_by_weight_desc",
                    description="Order by prop_gewicht DESC for heaviest requests.",
                    patterns=[re.compile(r"\border\s+by\s+prop_gewicht\s+desc\b", re.IGNORECASE)],
                )
            )

        if _POWER_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="prop_motor_leistung",
                    description="Use prop_motor_leistung for power-related questions.",
                    patterns=[re.compile(r"\bprop_motor_leistung\b", re.IGNORECASE)],
                )
            )

        if _WIDTH_RE.search(normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="width_columns",
                    description="Use width columns (prop_einbaubreite_max or prop_arbeitsbreite).",
                    patterns=[
                        re.compile(r"\bprop_einbaubreite_max\b", re.IGNORECASE),
                        re.compile(r"\bprop_arbeitsbreite\b", re.IGNORECASE),
                        re.compile(r"\bprop_einbaubreite_grundbohle\b", re.IGNORECASE),
                    ],
                )
            )

        if manufacturer_matches:
            for match in manufacturer_matches:
                code = (match.get("code") or "").strip()
                name = (match.get("name") or "").strip()
                if not (code or name):
                    continue
                patterns = []
                if code:
                    patterns.append(re.compile(rf"\bhersteller_code\s*=\s*'{re.escape(code)}'\b", re.IGNORECASE))
                    patterns.append(
                        re.compile(rf"\bhersteller_code\s+in\s*\([^)]*'{re.escape(code)}'[^)]*\)", re.IGNORECASE)
                    )
                if name:
                    patterns.append(
                        re.compile(rf"\bhersteller_name\s+ilike\s+'%{re.escape(name.lower())}%'", re.IGNORECASE)
                    )
                if patterns:
                    intent.required_constraints.append(
                        SQLConstraint(
                            name=f"manufacturer_{code or name}",
                            description=f"Filter by manufacturer {code or name}.",
                            patterns=patterns,
                        )
                    )

        if _FOLLOWUP_RE.search(normalized_query):
            ids = list((thread_state or {}).get("last_result_ids") or [])
            if ids:
                intent.followup_ids = ids
                intent.required_constraints.append(
                    SQLConstraint(
                        name="followup_ids",
                        description="Restrict to previous result ids.",
                        patterns=[re.compile(r"\bid\s+in\s*\(", re.IGNORECASE)],
                    )
                )

        if _TIME_RANGE_RE.search(normalized_query):
            date_columns = self._guess_date_columns()
            if date_columns:
                pattern = re.compile(
                    r"\b(" + "|".join(re.escape(col.lower()) for col in date_columns) + r")\b",
                    re.IGNORECASE,
                )
                intent.required_constraints.append(
                    SQLConstraint(
                        name="time_range",
                        description="Apply time range filter on a date column.",
                        patterns=[pattern],
                    )
                )
            else:
                intent.clarification = (
                    "Welches Datumsfeld soll ich fuer den Zeitraum verwenden (created, updated, oder ein anderes)?"
                )

        if _NULL_RE.search(normalized_query) and re.search(r"\bseriennummer\b", normalized_query):
            intent.required_constraints.append(
                SQLConstraint(
                    name="serial_null",
                    description="Handle missing serial numbers.",
                    patterns=[
                        re.compile(r"\bseriennummer\s+is\s+null\b", re.IGNORECASE),
                        re.compile(r"\bseriennummer\s*=\s*''", re.IGNORECASE),
                    ],
                )
            )

        if self._should_clarify(normalized_query, intent, thread_state):
            intent.clarification = (
                "Bitte konkretisieren: Maschinentyp, Hersteller oder Miete/Verkauf?"
            )

        return intent

    def _should_clarify(
        self,
        normalized_query: str,
        intent: SQLIntent,
        thread_state: Optional[Dict[str, Any]],
    ) -> bool:
        if intent.followup_ids or (thread_state or {}).get("last_result_ids"):
            return False
        if intent.request_type == "count":
            return False
        if _DOC_RE.search(normalized_query):
            return False
        if _RENTAL_RE.search(normalized_query) or _SALES_RE.search(normalized_query):
            return False
        if _AVAIL_RE.search(normalized_query):
            return False
        if _AC_RE.search(normalized_query) or _WEIGHT_RE.search(normalized_query):
            return False
        if _POWER_RE.search(normalized_query) or _WIDTH_RE.search(normalized_query):
            return False
        if re.search(r"\b(seriennummer|inventarnummer|hersteller)\b", normalized_query):
            return False
        return bool(re.search(r"\b(maschinen|gerate|geraete|equipment|devices)\b", normalized_query))

    def build_policy_message(self, intent: SQLIntent) -> Optional[str]:
        if not intent.required_constraints and not intent.followup_ids:
            return None
        lines = ["SQL POLICY (must follow):"]
        for constraint in intent.required_constraints:
            lines.append(f"- {constraint.description}")
        if intent.followup_ids:
            lines.append("- This is a follow-up; restrict to the previous ids with WHERE id IN (...).")
            preview_ids = ", ".join(str(value) for value in intent.followup_ids[:25])
            lines.append(f"- Allowed ids: {preview_ids}")
        return "\n".join(lines)

    def validate_sql(self, sql: str, intent: SQLIntent) -> SQLValidationResult:
        normalized_sql = _normalize_sql(sql)
        result = SQLValidationResult(ok=True, normalized_sql=normalized_sql)

        tables, columns = self._extract_tables_and_columns(sql)
        result.referenced_tables = tables
        result.referenced_columns = columns

        allowed_tables = set()
        if self.equipment_table:
            allowed_tables.add(self.equipment_table.lower())
            if "." in self.equipment_table:
                allowed_tables.add(self.equipment_table.split(".")[-1].lower())
        for table in tables:
            lowered = table.lower()
            if allowed_tables and lowered not in allowed_tables and not lowered.startswith("information_schema."):
                result.errors.append(f"Table not allowed: {table}")

        for constraint in intent.required_constraints:
            if not constraint.is_satisfied(normalized_sql):
                result.errors.append(f"Missing constraint: {constraint.name}")

        limit_value = self._extract_limit(normalized_sql)
        if limit_value is not None:
            result.limit_value = limit_value
            if intent.requested_limit and limit_value != intent.requested_limit:
                result.errors.append(
                    f"Incorrect limit: expected {intent.requested_limit}, got {limit_value}"
                )
        elif intent.requested_limit:
            result.errors.append(f"Missing LIMIT {intent.requested_limit}")

        if intent.request_type == "count" and "count(" not in normalized_sql:
            result.errors.append("Count query missing COUNT().")

        if self._column_resolver:
            allowed_columns = set(self._get_columns().keys())
            if allowed_columns:
                unknown_cols = [col for col in columns if col not in allowed_columns]
                if unknown_cols:
                    result.warnings.append(f"Unknown columns: {', '.join(sorted(unknown_cols))}")

        if result.errors:
            result.ok = False
        return result

    @staticmethod
    def _extract_limit(normalized_sql: str) -> Optional[int]:
        match = re.search(r"\blimit\s+(\d+)\b", normalized_sql)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        match = re.search(r"\bfetch\s+first\s+(\d+)\b", normalized_sql)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _extract_tables_and_columns(self, sql: str) -> Tuple[List[str], List[str]]:
        if SQLGLOT_AVAILABLE:
            try:
                parsed = sqlglot.parse_one(sql, read="postgres")
                tables = []
                columns = []
                for table in parsed.find_all(exp.Table):
                    tables.append(table.name or table.sql(dialect="postgres"))
                for column in parsed.find_all(exp.Column):
                    col_name = column.name
                    if col_name:
                        columns.append(col_name)
                return tables, columns
            except Exception:
                pass
        tables = self._extract_tables_regex(sql)
        columns = self._extract_columns_regex(sql)
        return tables, columns

    @staticmethod
    def _extract_tables_regex(sql: str) -> List[str]:
        matches = re.findall(r"\bfrom\s+([a-zA-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
        joins = re.findall(r"\bjoin\s+([a-zA-Z0-9_\.]+)", sql, flags=re.IGNORECASE)
        tables = matches + joins
        return [table.strip() for table in tables if table.strip()]

    @staticmethod
    def _extract_columns_regex(sql: str) -> List[str]:
        matches = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", sql)
        keywords = {
            "select",
            "from",
            "where",
            "and",
            "or",
            "group",
            "by",
            "order",
            "limit",
            "offset",
            "join",
            "on",
            "as",
            "count",
            "distinct",
            "ilike",
            "like",
            "is",
            "null",
            "true",
            "false",
            "in",
            "with",
            "union",
            "all",
            "case",
            "when",
            "then",
            "else",
            "end",
        }
        columns = []
        for token in matches:
            lowered = token.lower()
            if lowered in keywords or token.isdigit():
                continue
            columns.append(token)
        return columns
