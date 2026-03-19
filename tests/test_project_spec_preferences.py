import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.search import RAGSearch


def _search_without_init() -> RAGSearch:
    return RAGSearch.__new__(RAGSearch)


def test_user_preferences_override_assistant_option_text():
    search = _search_without_init()
    assistant_mcq = (
        "1) Welche Projektart liegt vor?\n"
        "A) Neubau\nB) Sanierung\nC) Erweiterung\nD) Sonstiges\n"
        "2) Welche Bauweise ist geplant?\n"
        "A) Asphalt\nB) Pflaster\nC) Schotter\nD) Beton\n"
        "3) Wie ist der Untergrund aktuell?\n"
        "A) Erdreich / ungebunden\nB) Bestehende Tragschicht\nC) Sehr hart / felsig\nD) Noch unklar\n"
        "4) Welche Belastung wird erwartet?\n"
        "A) Fussgaenger / Fahrrad\nB) PKW\nC) LKW / schwere Last\nD) Gemischt / unklar"
    )

    spec = search._extract_project_spec(
        query="Sanierung, Asphalt. Bestehende Tragschicht, gemischt / unklar",
        conversation_history=[{"role": "assistant", "content": assistant_mcq}],
        compound_response="",
        project_memory=None,
    )

    assert spec.project_type == "Sanierung"
    assert spec.construction_method == "Asphalt"
    assert spec.ground_condition == "Bestehende Tragschicht"
    assert spec.load_profile == "Gemischt / unklar"
    assert spec.delivery_preference == ""


def test_compact_width_phrase_is_detected():
    search = _search_without_init()
    spec = search._extract_project_spec(
        query="Empfehle mir bitte eine Maschine fuer den Asphalteinbau von 3,5m",
        conversation_history=[],
        compound_response="",
        project_memory=None,
    )

    assert spec.width_m == 3.5
    assert spec.length_m is None


def test_round_one_letter_answers_are_mapped():
    search = _search_without_init()
    spec = search._extract_project_spec(
        query="B, A, B, D",
        conversation_history=[],
        compound_response="",
        project_memory=None,
    )

    assert spec.project_type == "Sanierung"
    assert spec.construction_method == "Asphalt"
    assert spec.ground_condition == "Bestehende Tragschicht"
    assert spec.load_profile == "Gemischt / unklar"
