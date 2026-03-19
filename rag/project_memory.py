"""
Persistent project memory storage for advisory flows.

Stores compact project summaries per thread with a hard cap to keep
follow-up prompts small and predictable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ProjectMemoryStore:
    """Store up to N project memories per thread (Redis + in-memory fallback)."""

    redis_client: Any = None
    max_memories: int = 5
    max_age_seconds: int = 12 * 3600

    def __post_init__(self):
        self._fallback_memories: Dict[str, List[Dict[str, Any]]] = {}

    def _key(self, thread_key: str) -> str:
        return f"project_memories:{thread_key}"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _parse_created_at(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None

    def _is_fresh(self, entry: Dict[str, Any]) -> bool:
        created_at = self._parse_created_at(entry.get("created_at"))
        if not created_at:
            return False

        now_local = datetime.now().astimezone()
        created_local = created_at.astimezone()
        if created_local.date() != now_local.date():
            return False

        age_seconds = (now_local - created_local).total_seconds()
        return age_seconds <= max(1, int(self.max_age_seconds))

    def _normalize(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned: List[Dict[str, Any]] = []
        for item in memories:
            if isinstance(item, dict) and self._is_fresh(item):
                cleaned.append(item)
        return cleaned[-max(1, int(self.max_memories)) :]

    async def list_memories(self, thread_key: Optional[str]) -> List[Dict[str, Any]]:
        """Return all memories for a thread, newest last."""
        if not thread_key:
            return []

        if self.redis_client:
            try:
                raw = await self.redis_client.get(self._key(thread_key))
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        normalized = self._normalize(parsed)
                        if normalized != parsed:
                            if normalized:
                                await self.redis_client.setex(
                                    self._key(thread_key),
                                    max(1, int(self.max_age_seconds)),
                                    json.dumps(normalized, ensure_ascii=False),
                                )
                            else:
                                await self.redis_client.delete(self._key(thread_key))
                        return normalized
            except Exception:
                pass

        fallback = self._fallback_memories.get(thread_key, [])
        return self._normalize(fallback)

    async def latest_memory(self, thread_key: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the most recent memory for a thread."""
        memories = await self.list_memories(thread_key)
        return memories[-1] if memories else None

    async def add_memory(
        self,
        thread_key: Optional[str],
        summary: str,
        machine_rows: Optional[List[Dict[str, Any]]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append a memory entry and prune older entries (max_memories)."""
        if not thread_key:
            return None

        entry = {
            "created_at": self._now_iso(),
            "summary": (summary or "").strip()[:2000],
            "machine_rows": machine_rows or [],
            "meta": meta or {},
        }

        memories = await self.list_memories(thread_key)
        memories.append(entry)
        memories = self._normalize(memories)

        if self.redis_client:
            try:
                await self.redis_client.setex(
                    self._key(thread_key),
                    max(1, int(self.max_age_seconds)),
                    json.dumps(memories, ensure_ascii=False),
                )
                return entry
            except Exception:
                pass

        self._fallback_memories[thread_key] = memories
        return entry

    async def clear(self, thread_key: Optional[str]) -> None:
        """Clear memories for a thread."""
        if not thread_key:
            return

        if self.redis_client:
            try:
                await self.redis_client.delete(self._key(thread_key))
            except Exception:
                pass

        self._fallback_memories.pop(thread_key, None)
