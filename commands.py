"""Slash command handlers for the Teams bot."""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from rag.admin_logger import admin_logger
from rag.config import config
from rag.feedback import feedback_service

logger = logging.getLogger(__name__)

ReplyFn = Callable[[dict, str], Awaitable[None]]


def _canonical_command(command: str) -> str:
    cmd = (command or "").strip().lower()
    return COMMAND_ALIASES.get(cmd, cmd)


async def handle_legacy_document_command(body: dict, command_name: str, send_reply_func: ReplyFn):
    """Explain that the old vector-store document commands are no longer active."""
    message = (
        f"`{command_name}` ist nicht mehr aktiv.\n\n"
        "Die OpenAI-Vector-Store-Befehle wurden entfernt. "
        "Stelle stattdessen normale Fragen zu Maschinen oder Dokumenten direkt im Chat. "
        "Fuer Uploads oder Datenpflege bitte den Administrator verwenden."
    )
    await send_reply_func(body, message)


async def handle_reset_command(body: dict, args: list[str], send_reply_func: ReplyFn):
    """Confirm conversation reset."""
    _ = args
    await send_reply_func(
        body,
        (
            "**Verlauf zurueckgesetzt.**\n\n"
            "Die naechste Nachricht startet ohne bisherigen Projekt- oder Chatkontext."
        ),
    )


async def handle_help_command(body: dict, args: list[str], send_reply_func: ReplyFn):
    """Show active commands."""
    _ = args
    help_text = f"""
**RUEKO AI Assistant - Hilfe**

Aktive Befehle:
- `/hilfe` oder `/help` - diese Hilfe
- `/status` - Laufzeitstatus anzeigen
- `/zuruecksetzen` oder `/reset` - Verlauf fuer diesen Chat loeschen
- `/feedback <text>` - Feedback zur letzten Antwort speichern

Hinweise:
- Normale Fragen ohne Slash gehen direkt an Gemini + LangGraph.
- Bestandsfragen werden ueber SQL, PostgreSQL und Pinecone beantwortet.
- Projektberatung laeuft ueber Gemini mit Such-Unterstuetzung.
- Verlauf wird fuer ca. {config.conversation_ttl_hours} Stunden gehalten.

Entfernt:
- Die alten Dokument-Befehle wie `/liste`, `/suchen` oder `/hochladen` wurden aus dem aktiven Runtime-Pfad entfernt.
"""
    await send_reply_func(body, help_text.strip())


async def handle_status_command(body: dict, args: list[str], send_reply_func: ReplyFn):
    """Show active runtime configuration."""
    _ = args
    pinecone_summary = "nicht verfuegbar"

    try:
        from rag.vector_store import PineconeStore

        store = PineconeStore()
        stats = await store.get_stats()
        total_vectors = stats.get("total_vectors")
        if isinstance(total_vectors, int):
            pinecone_summary = f"{total_vectors} Vektoren"
        elif "error" in stats:
            pinecone_summary = f"Fehler: {stats['error']}"
    except Exception as exc:
        pinecone_summary = f"Fehler: {exc}"

    status_text = f"""
**System-Status**

- Advisory / Gemini: `{config.advisory_model}`
- LangGraph Retrieval: `{config.langgraph_model}`
- Fallback-Modell: `{config.fallback_model}`
- LangGraph aktiv: `{'ja' if config.use_langgraph_agent else 'nein'}`
- Advisory aktiv: `{'ja' if config.enable_compound_agent else 'nein'}`
- Pinecone Namespace: `{config.pinecone_namespace}`
- Pinecone Status: {pinecone_summary}
- Verlauf-TTL: `{config.conversation_ttl_hours}h`
"""
    await send_reply_func(body, status_text.strip())


async def handle_feedback_command(body: dict, args: list[str], send_reply_func: ReplyFn):
    """Store user feedback for the most recent conversation."""
    if not args:
        await send_reply_func(
            body,
            (
                "Bitte gib dein Feedback direkt nach dem Befehl an.\n\n"
                "Beispiel: `/feedback Die Antwort war hilfreich.`"
            ),
        )
        return

    user_id = body.get("from", {}).get("id", "unknown")
    feedback_text = " ".join(args).strip()

    try:
        feedback_saved = feedback_service.add_feedback(user_id=user_id, feedback=feedback_text)
        admin_saved = admin_logger.add_feedback(ms_user_id=user_id, feedback=feedback_text)

        if feedback_saved or admin_saved:
            await send_reply_func(
                body,
                (
                    "**Feedback gespeichert.**\n\n"
                    f"Rueckmeldung: _{feedback_text}_"
                ),
            )
            return

        await send_reply_func(
            body,
            (
                "Feedback konnte nicht gespeichert werden.\n\n"
                "Stelle zuerst eine Frage an den Bot und gib danach Feedback."
            ),
        )
    except Exception as exc:
        logger.error("Error in /feedback command: %s", exc, exc_info=True)
        await send_reply_func(body, f"Fehler beim Speichern des Feedbacks: {exc}")


COMMAND_HANDLERS = {
    "/hilfe": handle_help_command,
    "/status": handle_status_command,
    "/zuruecksetzen": handle_reset_command,
    "/feedback": handle_feedback_command,
}

LEGACY_COMMANDS = {
    "/hochladen",
    "/liste",
    "/loeschen",
    "/l\u00f6schen",
    "/suchen",
}

COMMAND_ALIASES = {
    "/help": "/hilfe",
    "/reset": "/zuruecksetzen",
    "/zur\u00fccksetzen": "/zuruecksetzen",
    "/rueckmeldung": "/feedback",
    "/r\u00fcckmeldung": "/feedback",
    "/upload": "/hochladen",
    "/list": "/liste",
    "/delete": "/loeschen",
    "/search": "/suchen",
}


async def handle_command(body: dict, command: str, send_reply_func: ReplyFn):
    """Route slash commands to active handlers."""
    parts = command.strip().split()
    if not parts:
        await send_reply_func(body, "Leerer Befehl.")
        return

    raw_cmd = parts[0].lower()
    cmd = _canonical_command(raw_cmd)
    args = parts[1:]

    if cmd in LEGACY_COMMANDS:
        logger.info("Legacy command rejected: %s", raw_cmd)
        await handle_legacy_document_command(body, raw_cmd, send_reply_func)
        return

    handler = COMMAND_HANDLERS.get(cmd)
    if handler:
        logger.info("Executing command: %s with args=%s", cmd, args)
        await handler(body, args, send_reply_func)
        return

    await send_reply_func(
        body,
        (
            f"Unbekannter Befehl: `{raw_cmd}`\n\n"
            "Tippe `/hilfe` fuer die aktiven Befehle. "
            "Normale Fragen ohne Slash gehen direkt an den Assistenten."
        ),
    )
