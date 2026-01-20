"""
German Commands Handler for RÜKO Teams Bot
Handles slash commands for document management and bot interaction.
"""
import os
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rag.feedback import feedback_service
from rag.admin_logger import admin_logger

load_dotenv()

logger = logging.getLogger(__name__)

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def handle_hochladen_command(body: dict, args: list, send_reply_func):
    """Handle /hochladen (upload) command - Currently disabled"""
    message = """
📤 **Dokument hochladen** - Bald verfügbar!

Diese Funktion wird in Kürze freigeschaltet.

**Grund:** Die Datei-Upload-Berechtigung muss vom Teams-Administrator aktiviert werden.

**Aktuell:** Dokumente werden manuell vom Administrator hochgeladen.

**So geht es weiter:**
1. Sende deine Dokumente per E-Mail an den Administrator
2. Administrator lädt sie in die Wissensdatenbank hoch
3. Du kannst sofort Fragen dazu stellen!

**Oder:** Warte bis die Upload-Funktion aktiviert ist (in Arbeit).

📧 Bei Fragen: Wende dich an den Administrator
"""

    await send_reply_func(body, message)


async def handle_liste_command(body: dict, args: list, send_reply_func):
    """Handle /liste (list) command - Show all documents"""
    try:
        # Get vector store files
        vector_store = await client.beta.vector_stores.retrieve(VECTOR_STORE_ID)
        files_response = await client.beta.vector_stores.files.list(VECTOR_STORE_ID)

        files = files_response.data

        if not files:
            await send_reply_func(body, "📭 Keine Dokumente in der Wissensdatenbank gefunden.")
            return

        # Format file list
        file_list = "📚 **Dokumente in der Wissensdatenbank:**\n\n"

        for idx, file in enumerate(files, 1):
            # Get file details
            file_obj = await client.files.retrieve(file.id)
            file_name = file_obj.filename
            file_size_kb = file_obj.bytes / 1024
            created_timestamp = file_obj.created_at

            # Format created date
            from datetime import datetime
            created_date = datetime.fromtimestamp(created_timestamp).strftime("%d.%m.%Y %H:%M")

            file_list += f"{idx}. **{file_name}**\n"
            file_list += f"   📊 Größe: {file_size_kb:.1f} KB\n"
            file_list += f"   📅 Hochgeladen: {created_date}\n"
            file_list += f"   🆔 ID: `{file.id[:25]}...`\n\n"

        file_list += f"\n💾 **Gesamt:** {len(files)} Dokumente\n"
        file_list += f"📦 Vector Store: `{VECTOR_STORE_ID}`"

        await send_reply_func(body, file_list)

    except Exception as e:
        logger.error(f"Error in /liste command: {e}")
        await send_reply_func(body, f"❌ Fehler beim Abrufen der Dokumentenliste: {str(e)}")


async def handle_löschen_command(body: dict, args: list, send_reply_func):
    """Handle /löschen (delete) command"""
    message = """
🗑️ **Dokument löschen** - Administrator-Funktion

Das Löschen von Dokumenten erfordert Administrator-Rechte.

**Aktuell:** Diese Funktion ist noch nicht aktiviert.

**Grund:** Sicherheitsmaßnahme - nur Administratoren dürfen Dokumente aus der Wissensdatenbank entfernen.

**So geht es:**
Wende dich an den Administrator, um ein Dokument zu löschen.

📧 Administrator kontaktieren für:
- Dokument löschen
- Dokument ersetzen
- Wissensdatenbank verwalten
"""

    await send_reply_func(body, message)


async def handle_suchen_command(body: dict, args: list, send_reply_func):
    """Handle /suchen (search) command"""
    if not args:
        await send_reply_func(body, "❌ Bitte gib einen Suchbegriff an:\n\n**Beispiel:** /suchen urlaub")
        return

    search_term = " ".join(args).lower()

    try:
        files_response = await client.beta.vector_stores.files.list(VECTOR_STORE_ID)
        files = files_response.data

        matches = []
        for file in files:
            file_obj = await client.files.retrieve(file.id)
            if search_term in file_obj.filename.lower():
                matches.append(file_obj)

        if not matches:
            await send_reply_func(body,
                f"🔍 Keine Dokumente gefunden für: **{search_term}**\n\n"
                f"💡 Tipp: Versuche andere Suchbegriffe oder nutze /liste um alle Dokumente anzuzeigen."
            )
            return

        result = f"🔍 **Suchergebnisse für '{search_term}':**\n\n"
        result += f"Gefunden: {len(matches)} Dokument(e)\n\n"

        for idx, file in enumerate(matches, 1):
            result += f"{idx}. **{file.filename}**\n"
            result += f"   🆔 ID: `{file.id[:25]}...`\n\n"

        await send_reply_func(body, result)

    except Exception as e:
        logger.error(f"Error in /suchen command: {e}")
        await send_reply_func(body, f"❌ Fehler bei der Suche: {str(e)}")


async def handle_zurücksetzen_command(body: dict, args: list, send_reply_func):
    """Handle /zurücksetzen (reset) command"""
    # Note: Actual conversation reset is handled by app.py when this message is processed
    # This command just confirms the reset to the user
    message = """🔄 **Gesprächsverlauf wird zurückgesetzt!**

Deine nächste Nachricht startet eine neue Unterhaltung.

💡 **Tipp:** Der Bot merkt sich den Kontext für 24 Stunden.
Nach einem Reset beginnt der Bot ohne Vorwissen über frühere Fragen."""

    await send_reply_func(body, message)


async def handle_hilfe_command(body: dict, args: list, send_reply_func):
    """Handle /hilfe (help) command"""
    help_text = """
📖 **RÜKO AI Assistant - Hilfe**

**📁 Dokumentenverwaltung:**

• **/hochladen** 📤
  Dokument hochladen (bald verfügbar)
  Aktuell: Nur durch Administrator

• **/liste** 📚
  Alle Dokumente in der Wissensdatenbank anzeigen
  Zeigt: Name, Größe, Upload-Datum, ID

• **/löschen** 🗑️
  Dokument löschen (Administrator-Funktion)

• **/suchen <begriff>** 🔍
  Dokumente nach Namen durchsuchen
  Beispiel: `/suchen urlaub`

**💬 Unterhaltung:**

• **/zurücksetzen** 🔄
  Gesprächsverlauf zurücksetzen
  Startet eine neue Konversation

• **/feedback <text>** 💬
  Feedback zur letzten Antwort geben
  Beispiel: `/feedback Sehr hilfreiche Antwort!`

**ℹ️ Information:**

• **/status** ℹ️
  System-Status und Statistiken anzeigen

• **/hilfe** ❓
  Diese Hilfe anzeigen

**💡 Tipps:**
• Stelle normale Fragen OHNE `/` für KI-Antworten
• Der Bot durchsucht automatisch alle Dokumente
• Bilder in PDFs werden mit KI analysiert
• Unterstützte Formate: PDF, DOCX, XLSX, JSON, CSV, TXT

**🤖 Über mich:**
Ich bin ein KI-Assistent für RÜKO-Dokumente.
Ich nutze OpenAI GPT-4o und durchsuche die Wissensdatenbank,
um präzise Antworten auf deine Fragen zu geben.

**Fragen?** Stell sie einfach direkt - ohne Befehle! 😊
"""

    await send_reply_func(body, help_text)


async def handle_status_command(body: dict, args: list, send_reply_func):
    """Handle /status command"""
    try:
        # Get vector store info
        vector_store = await client.beta.vector_stores.retrieve(VECTOR_STORE_ID)
        files_response = await client.beta.vector_stores.files.list(VECTOR_STORE_ID)
        file_count = len(files_response.data)

        status_text = f"""
📊 **System-Status**

**🤖 KI-Modell:**
• Modell: {OPENAI_MODEL}
• Anbieter: OpenAI
• Streaming: Aktiviert ✅

**📦 Wissensdatenbank:**
• Vector Store ID: `{VECTOR_STORE_ID}`
• Dokumente: {file_count}
• Status: {vector_store.status}

**💬 Konversationen:**
• Kontext-Speicher: Redis (24h TTL)

**✨ Funktionen:**
• 📚 Dokumentensuche
• 🖼️ Bild-Analyse (GPT-4o Vision)
• 💬 Konversations-Kontext
• ⚡ Echtzeit-Streaming

**✅ System betriebsbereit**

_Zuletzt geprüft: jetzt_
"""

        await send_reply_func(body, status_text)

    except Exception as e:
        logger.error(f"Error in /status command: {e}")
        await send_reply_func(body, f"❌ Fehler beim Abrufen des Status: {str(e)}")


async def handle_feedback_command(body: dict, args: list, send_reply_func):
    """Handle /feedback command - Store user feedback for their most recent conversation"""
    # Check if feedback text was provided
    if not args:
        await send_reply_func(
            body,
            "❌ **Feedback fehlt!**\n\n"
            "Bitte gib dein Feedback nach dem Befehl ein:\n\n"
            "**Beispiel:** `/feedback Die Antwort war sehr hilfreich!`\n\n"
            "💡 Dein Feedback hilft uns, den Bot zu verbessern."
        )
        return

    # Get user ID
    user_id = body.get("from", {}).get("id", "unknown")
    feedback_text = " ".join(args)

    try:
        # Store feedback in both systems (feedback_service is optional, admin_logger for dashboard)
        feedback_success = feedback_service.add_feedback(user_id=user_id, feedback=feedback_text)
        admin_success = admin_logger.add_feedback(ms_user_id=user_id, feedback=feedback_text)

        # Success if either system stored the feedback
        if feedback_success or admin_success:
            await send_reply_func(
                body,
                "**Danke für dein Feedback!**\n\n"
                f"Dein Feedback: _{feedback_text}_\n\n"
                "Deine Rückmeldung hilft uns, den Bot zu verbessern."
            )

            # Extract and save learned rule from feedback (async, non-blocking)
            try:
                from rag.learned_rules import learned_rules_service

                logger.info(f"[RuleExtraction] Starting for user {user_id[:20]}...")

                recent = admin_logger.get_most_recent_conversation(user_id)
                if not recent:
                    logger.warning(f"[RuleExtraction] No recent conversation found for user {user_id[:20]}")
                else:
                    logger.info(f"[RuleExtraction] Found context: Q='{recent['user_question'][:50]}...'")

                    rule = await learned_rules_service.extract_rule_from_feedback(
                        question=recent['user_question'],
                        response=recent['assistant_response'],
                        feedback=feedback_text
                    )

                    if not rule:
                        logger.info(f"[RuleExtraction] No rule extracted (LLM returned None)")
                    elif not rule.get('is_actionable'):
                        logger.info(f"[RuleExtraction] Rule not actionable: {rule.get('rule_text', 'N/A')[:50]}")
                    else:
                        saved = learned_rules_service.save_rule(rule)
                        if saved:
                            logger.info(f"[RuleExtraction] SUCCESS - Rule saved: {rule.get('rule_text', '')[:50]}...")
                        else:
                            logger.warning(f"[RuleExtraction] Rule extracted but save FAILED (duplicate or DB error)")
            except Exception as rule_error:
                # Don't fail the feedback command if rule extraction fails
                logger.warning(f"[RuleExtraction] Exception: {rule_error}")
        else:
            await send_reply_func(
                body,
                "**Feedback konnte nicht gespeichert werden.**\n\n"
                "Mögliche Gründe:\n"
                "- Kein vorheriges Gespräch gefunden\n"
                "- Du hast bereits Feedback zur letzten Antwort gegeben\n\n"
                "Stelle zuerst eine Frage an den Bot und gib dann Feedback."
            )

    except Exception as e:
        logger.error(f"Error in /feedback command: {e}")
        await send_reply_func(body, f"❌ Fehler beim Speichern des Feedbacks: {str(e)}")


# Command routing map
COMMAND_HANDLERS = {
    "/hochladen": handle_hochladen_command,
    "/liste": handle_liste_command,
    "/löschen": handle_löschen_command,
    "/suchen": handle_suchen_command,
    "/zurücksetzen": handle_zurücksetzen_command,
    "/hilfe": handle_hilfe_command,
    "/status": handle_status_command,
    "/feedback": handle_feedback_command,
    "/rückmeldung": handle_feedback_command,
}

# English aliases (for compatibility)
COMMAND_ALIASES = {
    "/upload": "/hochladen",
    "/list": "/liste",
    "/delete": "/löschen",
    "/search": "/suchen",
    "/reset": "/zurücksetzen",
    "/help": "/hilfe",
}


async def handle_command(body: dict, command: str, send_reply_func):
    """
    Main command router
    Routes German commands to appropriate handlers
    """
    # Extract command and arguments
    parts = command.strip().split()
    cmd = parts[0].lower()
    args = parts[1:] if len(parts) > 1 else []

    # Map English to German if used
    if cmd in COMMAND_ALIASES:
        cmd = COMMAND_ALIASES[cmd]

    # Get handler
    handler = COMMAND_HANDLERS.get(cmd)

    if handler:
        logger.info(f"Executing command: {cmd} with args: {args}")
        await handler(body, args, send_reply_func)
    else:
        # Unknown command
        await send_reply_func(
            body,
            f"❌ Unbekannter Befehl: {cmd}\n\n"
            f"Tippe **/hilfe** für alle verfügbaren Befehle.\n\n"
            f"💡 Tipp: Stelle Fragen ohne `/` für KI-Antworten!"
        )
