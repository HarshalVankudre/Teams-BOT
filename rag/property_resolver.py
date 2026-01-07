"""
Dynamic Property Resolver for SEMA Equipment Database.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s*\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text)
    tokens = re.split(r"[\s\-_]+", normalized)
    return [t for t in tokens if t and len(t) > 1]


class PropertyResolver:
    def __init__(
        self,
        postgres_service: Optional[Any] = None,
        csv_path: Optional[Path] = None,
    ):
        self._exact_index: Dict[str, Tuple[str, str, str]] = {}
        self._token_index: Dict[str, List[Tuple[str, str, float]]] = {}
        self._all_properties: List[Tuple[str, str, str]] = []
        self._load(postgres_service, csv_path)

    def _load(self, postgres_service: Optional[Any], csv_path: Optional[Path]) -> None:
        loaded = False
        if postgres_service and getattr(postgres_service, "available", False):
            try:
                rows = postgres_service.execute_query(
                    "SELECT code, name FROM property_types ORDER BY code"
                )
                if rows:
                    for row in rows:
                        code = row.get("code", "")
                        name = row.get("name", "")
                        if code and name:
                            self._index_property(code, name)
                    loaded = True
                    print(f"[PropertyResolver] Loaded {len(rows)} properties from database")
            except Exception as e:
                print(f"[PropertyResolver] Database load failed: {e}")
        if not loaded and csv_path:
            self._load_from_csv(csv_path)

    def _load_from_csv(self, csv_path: Path) -> None:
        if not csv_path.exists():
            print(f"[PropertyResolver] CSV not found: {csv_path}")
            return
        try:
            with csv_path.open("r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    code = (row.get("code") or "").strip()
                    name = (row.get("name") or "").strip()
                    if code and name:
                        self._index_property(code, name)
                        count += 1
                print(f"[PropertyResolver] Loaded {count} properties from CSV")
        except Exception as e:
            print(f"[PropertyResolver] CSV load error: {e}")

    def _index_property(self, code: str, name: str) -> None:
        code_lower = code.lower()
        name_normalized = _normalize_text(name)
        column_name = f"prop_{code_lower}_{self._slugify(name)}"
        self._all_properties.append((column_name, name, code))
        self._exact_index[name_normalized] = (column_name, name, code)
        self._exact_index[code_lower] = (column_name, name, code)
        tokens = _tokenize(name)
        for token in tokens:
            if token not in self._token_index:
                self._token_index[token] = []
            self._token_index[token].append((column_name, name, 1.0 / max(len(tokens), 1)))

    def _slugify(self, text: str) -> str:
        normalized = _normalize_text(text)
        slug = re.sub(r"[^a-z0-9]+", "_", normalized)
        slug = slug.strip("_")
        return slug[:50] if slug else "unknown"

    def resolve(self, user_term: str) -> Optional[str]:
        if not user_term:
            return None
        normalized = _normalize_text(user_term)
        if normalized in self._exact_index:
            return self._exact_index[normalized][0]
        if user_term.lower().startswith("prop_"):
            for col, name, code in self._all_properties:
                if col.lower() == user_term.lower():
                    return col
        tokens = _tokenize(user_term)
        if tokens:
            candidates = self._find_by_tokens(tokens)
            if candidates:
                return candidates[0][0]
        best_match = self._fuzzy_match(normalized)
        if best_match:
            return best_match
        return None

    def resolve_with_info(self, user_term: str) -> Optional[Tuple[str, str, str]]:
        if not user_term:
            return None
        normalized = _normalize_text(user_term)
        if normalized in self._exact_index:
            return self._exact_index[normalized]
        tokens = _tokenize(user_term)
        if tokens:
            candidates = self._find_by_tokens(tokens)
            if candidates:
                col = candidates[0][0]
                for c, name, code in self._all_properties:
                    if c == col:
                        return (c, name, code)
        return None

    def _find_by_tokens(self, tokens: List[str]) -> List[Tuple[str, str, float]]:
        scores: Dict[str, float] = {}
        names: Dict[str, str] = {}
        for token in tokens:
            if token in self._token_index:
                for col, name, score in self._token_index[token]:
                    scores[col] = scores.get(col, 0) + score
                    names[col] = name
            else:
                for idx_token, matches in self._token_index.items():
                    if token in idx_token or idx_token in token:
                        similarity = SequenceMatcher(None, token, idx_token).ratio()
                        if similarity > 0.7:
                            for col, name, score in matches:
                                scores[col] = scores.get(col, 0) + score * similarity
                                names[col] = name
        sorted_results = sorted(scores.items(), key=lambda x: -x[1])
        return [(col, names.get(col, ""), score) for col, score in sorted_results[:5]]

    def _fuzzy_match(self, normalized_term: str, threshold: float = 0.6) -> Optional[str]:
        best_score = 0.0
        best_col = None
        for col, name, code in self._all_properties:
            name_normalized = _normalize_text(name)
            score = SequenceMatcher(None, normalized_term, name_normalized).ratio()
            if score > best_score and score >= threshold:
                best_score = score
                best_col = col
        return best_col

    def get_suggestions(self, user_term: str, limit: int = 5) -> List[Tuple[str, str, float]]:
        if not user_term:
            return []
        normalized = _normalize_text(user_term)
        results = []
        for col, name, code in self._all_properties:
            name_normalized = _normalize_text(name)
            score = SequenceMatcher(None, normalized, name_normalized).ratio()
            if score > 0.3:
                results.append((col, f"{name} ({code})", score))
        results.sort(key=lambda x: -x[2])
        return results[:limit]

    def list_all(self) -> List[Tuple[str, str, str]]:
        return list(self._all_properties)

    @property
    def count(self) -> int:
        return len(self._all_properties)


def create_property_resolver(postgres_service: Optional[Any] = None) -> PropertyResolver:
    csv_path = Path(__file__).resolve().parent.parent / "sql_export" / "property_types.csv"
    return PropertyResolver(postgres_service=postgres_service, csv_path=csv_path)
