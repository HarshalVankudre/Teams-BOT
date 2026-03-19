import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.project_memory import ProjectMemoryStore
from rag.search import RAGSearch


def _search_without_init() -> RAGSearch:
    return RAGSearch.__new__(RAGSearch)


def test_serial_number_property_lookup_stays_retrieval():
    search = _search_without_init()
    query = "Welche Hoehe hat die Maschine mit der Seriennummer: KBCEZN5BHSWA55197"

    assert search._is_machine_lookup_query(query)
    assert search._is_explicit_retrieval_only(query)
    assert not search._is_advisory_query(query)


def test_all_properties_lookup_stays_retrieval():
    search = _search_without_init()
    query = "Bitte gib mir alle Infos zur Maschine 101870941182 die du in der Datenbank finden kannst"

    assert search._is_machine_lookup_query(query)
    assert search._is_explicit_retrieval_only(query)
    assert not search._is_advisory_query(query)


def test_inventory_recommendation_stays_retrieval():
    search = _search_without_init()
    query = "Welche Maschine empfiehlst du mir aus dem Mietpark"

    assert search._is_inventory_recommendation_query(query)
    assert search._is_explicit_retrieval_only(query)
    assert not search._is_advisory_query(query)


def test_model_info_lookup_stays_retrieval():
    search = _search_without_init()
    query = "Gib mir naehere Infos zum Super 1300-3i"

    assert search._is_explicit_retrieval_only(query)
    assert not search._is_advisory_query(query)


def test_raw_machine_dump_query_is_detected():
    search = _search_without_init()
    query = "Bitte ohne Interpretation einfach nur Eigenschaft und Wert aus der Datenbank fuer Maschine 101870941182"

    assert search._is_raw_machine_dump_query(query)
    assert search._extract_machine_identifier(query) == ("reference", "101870941182")


def test_machine_property_is_extracted_from_row():
    search = _search_without_init()
    row = {
        "seriennummer": "KBCEZN5BHSWA55197",
        "nuclos_state": "Released",
        "hoehe_mm_num": 2540,
        "verwendung_code": "MIET",
        "verwendung_name": "Vermietung",
        "properties_jsonb": {"prop_e2970_bohle_typ": "AB 220-3 TV"},
    }

    height = search._extract_property_from_machine_row(
        row,
        "Welche Hoehe hat die Maschine mit der Seriennummer KBCEZN5BHSWA55197?",
    )
    usage = search._extract_property_from_machine_row(
        row,
        "Welche Verwendung hat die Maschine mit der Seriennummer KBCEZN5BHSWA55197?",
    )
    status = search._extract_property_from_machine_row(
        row,
        "Ist die Maschine mit der Seriennummer KBCEZN5BHSWA55197 verfuegbar?",
    )

    assert height == {"label": "Hoehe", "value": "2540"}
    assert usage == {"label": "Verwendung", "value": "Vermietung"}
    assert "Released" in status["value"]


def test_direct_machine_property_response_uses_postgres_direct():
    search = _search_without_init()
    row = {
        "seriennummer": "KBCEZN5BHSWA55197",
        "nuclos_state": "Released",
        "hoehe_mm_num": 2540,
        "properties_jsonb": {},
    }
    search._fetch_machine_full_row = lambda identifier_type, identifier_value: row
    search._get_conversation_history = lambda thread_key: []

    result = asyncio.run(
        search._try_direct_machine_property_response(
            "Welche Hoehe hat die Maschine mit der Seriennummer: KBCEZN5BHSWA55197",
            thread_key="test-thread",
        )
    )

    assert result["agent"] == "postgres_direct"
    assert "2540" in result["response"]


def test_project_recommendation_still_routes_to_advisory():
    search = _search_without_init()
    query = "Empfehle mir bitte eine Maschine fuer den Asphalteinbau von 3,5m"

    assert search._is_advisory_query(query)


def test_released_status_is_not_presented_as_live_availability():
    search = _search_without_init()

    assert search._format_availability("Released") == "Im Bestand (Released)"
    assert search._format_availability("Locked") == "Gesperrt (Locked)"


def test_stale_local_advisory_state_is_pruned():
    search = _search_without_init()
    thread_key = "thread-1"
    search._advisory_threads = {thread_key: 0.0}
    search._advisory_recommended_fallback = {thread_key: 0.0}
    search._history_fallback = {
        thread_key: {
            "updated_at": 0.0,
            "messages": [{"role": "user", "content": "altes projekt"}],
        }
    }

    search._prune_local_thread_state(thread_key)

    assert thread_key not in search._advisory_threads
    assert thread_key not in search._advisory_recommended_fallback
    assert thread_key not in search._history_fallback


def test_stale_project_memory_is_ignored():
    store = ProjectMemoryStore(redis_client=None, max_memories=5, max_age_seconds=12 * 3600)
    store._fallback_memories["thread-1"] = [
        {
            "created_at": "2000-01-01T00:00:00+00:00",
            "summary": "Altes Projekt",
            "machine_rows": [],
            "meta": {},
        }
    ]

    latest = asyncio.run(store.latest_memory("thread-1"))

    assert latest is None
