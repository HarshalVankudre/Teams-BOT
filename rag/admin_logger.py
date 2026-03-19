"""
Admin Logging Service (SQLite)

Logs Teams bot conversations to a local SQLite database for dashboard analytics.

Configure via env vars:
- ADMIN_SQLITE_PATH (optional, defaults to data/admin.db)
"""
import os
import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AdminConfig:
    """Admin database configuration."""
    db_path: str

    @classmethod
    def from_env(cls) -> "AdminConfig":
        default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "admin.db")
        return cls(
            db_path=os.getenv("ADMIN_SQLITE_PATH", default_path),
        )

    def validate(self) -> Optional[str]:
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                return f"Cannot create directory for admin DB: {e}"
        return None


class AdminLogger:
    """
    Logs conversations to local SQLite database for dashboard analytics.

    Usage:
        logger = AdminLogger()
        logger.log_message(
            thread_id="user123:conv456",
            ms_user_id="user123",
            user_name="John Doe",
            user_email="john@example.com",
            role="user",
            content="Wie viele Bagger haben wir?",
            response_time_ms=1500,
            tools_used=["execute_sql"],
            sql_query="SELECT COUNT(*) FROM equipment_matrix..."
        )
    """

    def __init__(self, config: Optional[AdminConfig] = None):
        self.config = config or AdminConfig.from_env()
        self._local = threading.local()
        self.available = False

        print(f"[AdminLogger] Config: db_path={self.config.db_path}")

        config_error = self.config.validate()
        if config_error:
            print(f"[AdminLogger] Disabled ({config_error})")
            return

        try:
            self._ensure_schema()
            self.available = True
            print(f"[AdminLogger] Connected to SQLite database: {self.config.db_path}")
        except Exception as e:
            print(f"[AdminLogger] Init failed: {e}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local SQLite connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.config.db_path, timeout=30)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    @contextmanager
    def _db(self):
        """Context manager for safe connection handling."""
        conn = self._get_conn()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ms_user_id TEXT NOT NULL UNIQUE,
                display_name TEXT,
                email TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL UNIQUE,
                user_id INTEGER,
                message_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_time_ms INTEGER,
                tools_used TEXT,
                sql_query TEXT,
                sql_results_count INTEGER,
                error TEXT,
                logs TEXT,
                feedback TEXT,
                feedback_at TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE TABLE IF NOT EXISTS learned_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_text TEXT NOT NULL,
                category TEXT,
                keywords TEXT,
                source_question TEXT,
                source_feedback TEXT,
                confidence_score REAL DEFAULT 1.0,
                usage_count INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_thread_id ON conversations(thread_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
            CREATE INDEX IF NOT EXISTS idx_messages_feedback ON messages(feedback);
            CREATE INDEX IF NOT EXISTS idx_rules_active ON learned_rules(is_active);
            CREATE INDEX IF NOT EXISTS idx_users_ms_user_id ON users(ms_user_id);
        """)
        conn.commit()

    def _get_or_create_user(self, conn, ms_user_id: str, display_name: str = None, email: str = None) -> int:
        """Get or create user and return user ID."""
        cur = conn.execute("SELECT id FROM users WHERE ms_user_id = ?", (ms_user_id,))
        row = cur.fetchone()

        if row:
            conn.execute(
                "UPDATE users SET last_active = datetime('now'), display_name = COALESCE(?, display_name), email = COALESCE(?, email) WHERE id = ?",
                (display_name, email, row[0])
            )
            return row[0]

        cur = conn.execute(
            "INSERT INTO users (ms_user_id, display_name, email) VALUES (?, ?, ?)",
            (ms_user_id, display_name, email)
        )
        return cur.lastrowid

    def _get_or_create_conversation(self, conn, thread_id: str, user_id: int) -> int:
        """Get or create conversation and return conversation ID."""
        cur = conn.execute("SELECT id FROM conversations WHERE thread_id = ?", (thread_id,))
        row = cur.fetchone()

        if row:
            conn.execute(
                "UPDATE conversations SET last_message_at = datetime('now'), message_count = message_count + 1 WHERE id = ?",
                (row[0],)
            )
            return row[0]

        cur = conn.execute(
            "INSERT INTO conversations (thread_id, user_id, message_count) VALUES (?, ?, 1)",
            (thread_id, user_id)
        )
        return cur.lastrowid

    def log_message(
        self,
        thread_id: str,
        ms_user_id: str,
        role: str,
        content: str,
        user_name: str = None,
        user_email: str = None,
        response_time_ms: int = None,
        tools_used: List[str] = None,
        sql_query: str = None,
        sql_results_count: int = None,
        error: str = None,
        logs: Dict[str, Any] = None
    ) -> bool:
        """Log a message to the admin database."""
        if not self.available:
            return False
        try:
            logs_json = json.dumps(logs) if logs else None
            tools_json = json.dumps(tools_used) if tools_used else None

            with self._db() as conn:
                user_id = self._get_or_create_user(conn, ms_user_id, user_name, user_email)
                conversation_id = self._get_or_create_conversation(conn, thread_id, user_id)
                conn.execute("""
                    INSERT INTO messages (
                        conversation_id, role, content, response_time_ms,
                        tools_used, sql_query, sql_results_count, error, logs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id, role, content, response_time_ms,
                    tools_json, sql_query, sql_results_count, error, logs_json
                ))
                conn.commit()
            return True

        except Exception as e:
            print(f"[AdminLogger] Error logging message: {e}")
            return False

    def log_conversation(
        self,
        thread_id: str,
        ms_user_id: str,
        user_message: str,
        assistant_response: str,
        user_name: str = None,
        user_email: str = None,
        response_time_ms: int = None,
        tools_used: List[str] = None,
        sql_query: str = None,
        sql_results_count: int = None,
        error: str = None,
        logs: Dict[str, Any] = None
    ) -> bool:
        """Log both user message and assistant response in a single transaction."""
        if not self.available:
            return False
        try:
            logs_json = json.dumps(logs) if logs else None
            tools_json = json.dumps(tools_used) if tools_used else None

            with self._db() as conn:
                user_id = self._get_or_create_user(conn, ms_user_id, user_name, user_email)
                conversation_id = self._get_or_create_conversation(conn, thread_id, user_id)

                conn.execute("""
                    INSERT INTO messages (
                        conversation_id, role, content, response_time_ms,
                        tools_used, sql_query, sql_results_count, error, logs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (conversation_id, 'user', user_message, None, None, None, None, None, None))

                conn.execute("""
                    INSERT INTO messages (
                        conversation_id, role, content, response_time_ms,
                        tools_used, sql_query, sql_results_count, error, logs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id, 'assistant', assistant_response, response_time_ms,
                    tools_json, sql_query, sql_results_count, error, logs_json
                ))

                conn.commit()
            return True
        except Exception as e:
            print(f"[AdminLogger] Error logging conversation: {e}")
            return False

    def add_feedback(self, ms_user_id: str, feedback: str) -> bool:
        """Add feedback to the most recent assistant message for a user."""
        if not self.available:
            return False
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    UPDATE messages SET feedback = ?, feedback_at = datetime('now')
                    WHERE id = (
                        SELECT m.id FROM messages m
                        JOIN conversations c ON m.conversation_id = c.id
                        JOIN users u ON c.user_id = u.id
                        WHERE u.ms_user_id = ?
                        AND m.role = 'assistant'
                        AND m.feedback IS NULL
                        ORDER BY m.created_at DESC
                        LIMIT 1
                    )
                """, (feedback, ms_user_id))

                conn.commit()

            if cur.rowcount > 0:
                print(f"[AdminLogger] Feedback added")
                return True
            else:
                print(f"[AdminLogger] No recent message found for user {ms_user_id[:20]}...")
                return False

        except Exception as e:
            print(f"[AdminLogger] Error adding feedback: {e}")
            return False

    def _rows_to_dicts(self, cursor) -> List[Dict[str, Any]]:
        """Convert cursor results to list of dicts."""
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_all_conversations(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all conversations with user info for admin dashboard."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                           u.display_name, u.email, u.ms_user_id,
                           (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_message,
                           (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) as feedback_count
                    FROM conversations c
                    JOIN users u ON c.user_id = u.id
                    ORDER BY c.last_message_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error getting conversations: {e}")
            return []

    def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a specific conversation."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT m.id, m.role, m.content, m.created_at, m.response_time_ms,
                           m.tools_used, m.sql_query, m.sql_results_count, m.error, m.feedback, m.feedback_at
                    FROM messages m
                    WHERE m.conversation_id = ?
                    ORDER BY m.created_at ASC
                """, (conversation_id,))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error getting messages: {e}")
            return []

    def get_conversation_with_user(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get conversation details with user info."""
        if not self.available:
            return None
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                           u.id as user_id, u.display_name, u.email, u.ms_user_id, u.first_seen, u.last_active
                    FROM conversations c
                    JOIN users u ON c.user_id = u.id
                    WHERE c.id = ?
                """, (conversation_id,))
                rows = self._rows_to_dicts(cur)
                return rows[0] if rows else None
        except Exception as e:
            print(f"[AdminLogger] Error getting conversation: {e}")
            return None

    def get_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all users for admin dashboard."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT u.id, u.ms_user_id, u.display_name, u.email, u.first_seen, u.last_active,
                           COUNT(DISTINCT c.id) as conversation_count,
                           COUNT(m.id) as message_count
                    FROM users u
                    LEFT JOIN conversations c ON u.id = c.user_id
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    GROUP BY u.id
                    ORDER BY u.last_active DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error getting users: {e}")
            return []

    def get_all_feedback(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all feedback entries for admin dashboard."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT m.id, m.content as assistant_response, m.feedback, m.feedback_at, m.created_at,
                           u.display_name, u.email,
                           (SELECT content FROM messages WHERE conversation_id = m.conversation_id AND role = 'user'
                            AND created_at < m.created_at ORDER BY created_at DESC LIMIT 1) as user_question
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    JOIN users u ON c.user_id = u.id
                    WHERE m.feedback IS NOT NULL
                    ORDER BY m.feedback_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error getting feedback: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        if not self.available:
            return {}
        try:
            with self._db() as conn:
                stats = {}

                cur = conn.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM conversations) as total_conversations,
                        (SELECT COUNT(*) FROM users) as total_users,
                        (SELECT COUNT(*) FROM messages) as total_messages,
                        (SELECT COUNT(*) FROM messages WHERE created_at >= date('now')) as messages_today,
                        (SELECT COUNT(*) FROM messages WHERE created_at >= date('now', '-7 days')) as messages_this_week,
                        (SELECT AVG(response_time_ms) FROM messages WHERE response_time_ms IS NOT NULL) as avg_response_time,
                        (SELECT COUNT(*) FROM messages WHERE feedback IS NOT NULL) as total_feedback
                """)
                row = cur.fetchone()
                stats['total_conversations'] = row[0]
                stats['total_users'] = row[1]
                stats['total_messages'] = row[2]
                stats['messages_today'] = row[3]
                stats['messages_this_week'] = row[4]
                stats['avg_response_time_ms'] = round(row[5]) if row[5] else 0
                stats['total_feedback'] = row[6]

                cur = conn.execute("""
                    SELECT c.id, u.display_name, c.last_message_at,
                           (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_message
                    FROM conversations c
                    JOIN users u ON c.user_id = u.id
                    ORDER BY c.last_message_at DESC
                    LIMIT 5
                """)
                stats['recent_conversations'] = [
                    {'id': row[0], 'display_name': row[1], 'last_message_at': row[2], 'first_message': row[3]}
                    for row in cur.fetchall()
                ]

                cur = conn.execute("""
                    SELECT m.id, u.display_name, m.feedback, m.feedback_at
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    JOIN users u ON c.user_id = u.id
                    WHERE m.feedback IS NOT NULL
                    ORDER BY m.feedback_at DESC
                    LIMIT 5
                """)
                stats['recent_feedback'] = [
                    {'id': row[0], 'display_name': row[1], 'feedback': row[2], 'feedback_at': row[3]}
                    for row in cur.fetchall()
                ]

            return stats
        except Exception as e:
            print(f"[AdminLogger] Error getting statistics: {e}")
            return {}

    def search_conversations(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search conversations by user name or message content."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                search_pattern = f"%{query}%"
                cur = conn.execute("""
                    SELECT DISTINCT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                           u.display_name, u.email
                    FROM conversations c
                    JOIN users u ON c.user_id = u.id
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE u.display_name LIKE ? OR u.email LIKE ? OR m.content LIKE ?
                    ORDER BY c.last_message_at DESC
                    LIMIT ?
                """, (search_pattern, search_pattern, search_pattern, limit))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error searching conversations: {e}")
            return []

    def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all conversations for a specific user."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                           (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_message
                    FROM conversations c
                    WHERE c.user_id = ?
                    ORDER BY c.last_message_at DESC
                    LIMIT ?
                """, (user_id, limit))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error getting user conversations: {e}")
            return []

    # ========== DELETE METHODS ==========

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a single conversation and all its messages."""
        if not self.available:
            return False
        try:
            with self._db() as conn:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
                conn.commit()
            return cur.rowcount > 0
        except Exception as e:
            print(f"[AdminLogger] Error deleting conversation: {e}")
            return False

    def delete_user_conversations(self, user_id: int) -> int:
        """Delete all conversations for a user. Returns count of deleted conversations."""
        if not self.available:
            return 0
        try:
            with self._db() as conn:
                cur = conn.execute("SELECT id FROM conversations WHERE user_id = ?", (user_id,))
                conv_ids = [row[0] for row in cur.fetchall()]

                if not conv_ids:
                    return 0

                placeholders = ",".join(["?"] * len(conv_ids))
                conn.execute(f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", conv_ids)
                conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                conn.commit()

            print(f"[AdminLogger] Deleted {len(conv_ids)} conversations for user #{user_id}")
            return len(conv_ids)
        except Exception as e:
            print(f"[AdminLogger] Error deleting user conversations: {e}")
            return 0

    def delete_user(self, user_id: int) -> bool:
        """Delete a user and all their conversations."""
        if not self.available:
            return False
        try:
            self.delete_user_conversations(user_id)
            with self._db() as conn:
                cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
                conn.commit()

            if cur.rowcount > 0:
                print(f"[AdminLogger] Deleted user #{user_id}")
                return True
            return False
        except Exception as e:
            print(f"[AdminLogger] Error deleting user: {e}")
            return False

    # ========== ADVANCED SEARCH METHODS ==========

    def search_users(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search users by name or email."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                search_pattern = f"%{query}%"
                cur = conn.execute("""
                    SELECT u.id, u.ms_user_id, u.display_name, u.email, u.first_seen, u.last_active,
                           COUNT(DISTINCT c.id) as conversation_count,
                           COUNT(m.id) as message_count
                    FROM users u
                    LEFT JOIN conversations c ON u.id = c.user_id
                    LEFT JOIN messages m ON c.id = m.conversation_id
                    WHERE u.display_name LIKE ? OR u.email LIKE ?
                    GROUP BY u.id
                    ORDER BY u.last_active DESC
                    LIMIT ?
                """, (search_pattern, search_pattern, limit))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error searching users: {e}")
            return []

    def search_feedback(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search feedback by user name or feedback text."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                search_pattern = f"%{query}%"
                cur = conn.execute("""
                    SELECT m.id, m.content as assistant_response, m.feedback, m.feedback_at, m.created_at,
                           u.display_name, u.email,
                           (SELECT content FROM messages WHERE conversation_id = m.conversation_id AND role = 'user'
                            AND created_at < m.created_at ORDER BY created_at DESC LIMIT 1) as user_question
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    JOIN users u ON c.user_id = u.id
                    WHERE m.feedback IS NOT NULL
                    AND (u.display_name LIKE ? OR m.feedback LIKE ?)
                    ORDER BY m.feedback_at DESC
                    LIMIT ?
                """, (search_pattern, search_pattern, limit))
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error searching feedback: {e}")
            return []

    def get_conversations_filtered(
        self,
        user_id: int = None,
        date_from: str = None,
        date_to: str = None,
        has_feedback: bool = None,
        search: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get conversations with advanced filters."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                conditions = []
                params = []

                if user_id:
                    conditions.append("c.user_id = ?")
                    params.append(user_id)

                if date_from:
                    conditions.append("c.created_at >= ?")
                    params.append(date_from)

                if date_to:
                    conditions.append("c.created_at <= ?")
                    params.append(date_to + " 23:59:59")

                if has_feedback is True:
                    conditions.append("(SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) > 0")
                elif has_feedback is False:
                    conditions.append("(SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) = 0")

                if search:
                    search_pattern = f"%{search}%"
                    conditions.append("(u.display_name LIKE ? OR EXISTS (SELECT 1 FROM messages WHERE conversation_id = c.id AND content LIKE ?))")
                    params.extend([search_pattern, search_pattern])

                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

                query = f"""
                    SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                           u.display_name, u.email, u.ms_user_id,
                           (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_message,
                           (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) as feedback_count
                    FROM conversations c
                    JOIN users u ON c.user_id = u.id
                    {where_clause}
                    ORDER BY c.last_message_at DESC
                    LIMIT ? OFFSET ?
                """
                params.extend([limit, offset])

                cur = conn.execute(query, params)
                return self._rows_to_dicts(cur)
        except Exception as e:
            print(f"[AdminLogger] Error getting filtered conversations: {e}")
            return []

    def get_all_users_simple(self) -> List[Dict[str, Any]]:
        """Get all users (id and name only) for dropdown filters."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("SELECT id, display_name FROM users ORDER BY display_name")
                return [{'id': row[0], 'display_name': row[1] or 'Unbekannt'} for row in cur.fetchall()]
        except Exception as e:
            print(f"[AdminLogger] Error getting users simple: {e}")
            return []

    # ========== LEARNED RULES METHODS ==========

    def get_most_recent_conversation(self, ms_user_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent Q&A pair for a user (for rule extraction context)."""
        if not self.available:
            return None
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT
                        m.content as assistant_response,
                        (SELECT content FROM messages
                         WHERE conversation_id = m.conversation_id
                         AND role = 'user'
                         AND created_at < m.created_at
                         ORDER BY created_at DESC
                         LIMIT 1) as user_question
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    JOIN users u ON c.user_id = u.id
                    WHERE u.ms_user_id = ?
                    AND m.role = 'assistant'
                    ORDER BY m.created_at DESC
                    LIMIT 1
                """, (ms_user_id,))

                row = cur.fetchone()

            if row and row[1]:
                return {
                    'assistant_response': row[0],
                    'user_question': row[1]
                }
            return None

        except Exception as e:
            print(f"[AdminLogger] Error getting recent conversation: {e}")
            return None

    def save_learned_rule(self, rule: Dict[str, Any]) -> bool:
        """Save a learned rule to the database."""
        if not self.available:
            return False
        try:
            keywords = rule.get('keywords', [])
            keywords_json = json.dumps(keywords) if isinstance(keywords, list) else keywords

            with self._db() as conn:
                cur = conn.execute("""
                    INSERT INTO learned_rules (
                        rule_text, category, keywords, source_question,
                        source_feedback, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    rule.get('rule_text'),
                    rule.get('category'),
                    keywords_json,
                    rule.get('source_question'),
                    rule.get('source_feedback'),
                    rule.get('confidence_score', 1.0)
                ))
                conn.commit()

            if cur.lastrowid:
                print(f"[AdminLogger] Saved learned rule #{cur.lastrowid}: {rule.get('rule_text', '')[:50]}...")
                return True
            return False

        except Exception as e:
            print(f"[AdminLogger] Error saving learned rule: {e}")
            return False

    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get all active learned rules."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    SELECT id, rule_text, category, keywords, confidence_score, usage_count, created_at
                    FROM learned_rules
                    WHERE is_active = 1
                    ORDER BY usage_count DESC, confidence_score DESC, created_at DESC
                """)
                return self._rows_to_dicts(cur)

        except Exception as e:
            print(f"[AdminLogger] Error getting active rules: {e}")
            return []

    def increment_rule_usage(self, rule_id: int) -> bool:
        """Increment the usage count for a rule."""
        if not self.available:
            return False
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    UPDATE learned_rules
                    SET usage_count = usage_count + 1
                    WHERE id = ?
                """, (rule_id,))
                conn.commit()
            return cur.rowcount > 0

        except Exception as e:
            print(f"[AdminLogger] Error incrementing rule usage: {e}")
            return False

    def get_all_rules(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """Get all learned rules (for admin dashboard)."""
        if not self.available:
            return []
        try:
            with self._db() as conn:
                if include_inactive:
                    cur = conn.execute("""
                        SELECT id, rule_text, category, keywords, source_question, source_feedback,
                               confidence_score, usage_count, is_active, created_at
                        FROM learned_rules
                        ORDER BY created_at DESC
                    """)
                else:
                    cur = conn.execute("""
                        SELECT id, rule_text, category, keywords, source_question, source_feedback,
                               confidence_score, usage_count, is_active, created_at
                        FROM learned_rules
                        WHERE is_active = 1
                        ORDER BY created_at DESC
                    """)
                return self._rows_to_dicts(cur)

        except Exception as e:
            print(f"[AdminLogger] Error getting all rules: {e}")
            return []

    def toggle_rule_active(self, rule_id: int, is_active: bool) -> bool:
        """Activate or deactivate a learned rule."""
        if not self.available:
            return False
        try:
            with self._db() as conn:
                cur = conn.execute("""
                    UPDATE learned_rules
                    SET is_active = ?
                    WHERE id = ?
                """, (1 if is_active else 0, rule_id))
                conn.commit()

            if cur.rowcount > 0:
                print(f"[AdminLogger] Rule #{rule_id} is_active set to {is_active}")
                return True
            return False

        except Exception as e:
            print(f"[AdminLogger] Error toggling rule active: {e}")
            return False

    def delete_rule(self, rule_id: int) -> bool:
        """Delete a learned rule."""
        if not self.available:
            return False
        try:
            with self._db() as conn:
                cur = conn.execute("DELETE FROM learned_rules WHERE id = ?", (rule_id,))
                conn.commit()

            if cur.rowcount > 0:
                print(f"[AdminLogger] Deleted rule #{rule_id}")
                return True
            return False

        except Exception as e:
            print(f"[AdminLogger] Error deleting rule: {e}")
            return False


# Global instance
admin_logger = AdminLogger()
