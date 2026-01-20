"""
Admin Logging Service

Logs Teams bot conversations to an admin database (optional).

Configure via env vars (recommended separate DB):
- ADMIN_POSTGRES_DB (required to enable)
- ADMIN_POSTGRES_PASSWORD (required to enable)
- ADMIN_POSTGRES_HOST / ADMIN_POSTGRES_PORT / ADMIN_POSTGRES_USER (optional overrides)

Falls back to POSTGRES_HOST/POSTGRES_PORT/POSTGRES_USER if the ADMIN_* variants
aren't set.
"""
import os

try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[WARNING] psycopg2 not installed. Admin logging disabled.")

from typing import Optional, List, Dict, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AdminConfig:
    """Admin database configuration."""
    host: str
    port: str
    database: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "AdminConfig":
        return cls(
            host=os.getenv("ADMIN_POSTGRES_HOST") or os.getenv("POSTGRES_HOST", ""),
            port=os.getenv("ADMIN_POSTGRES_PORT") or os.getenv("POSTGRES_PORT", ""),
            database=os.getenv("ADMIN_POSTGRES_DB", ""),
            user=os.getenv("ADMIN_POSTGRES_USER") or os.getenv("POSTGRES_USER", ""),
            password=os.getenv("ADMIN_POSTGRES_PASSWORD", ""),
        )

    def validate(self) -> Optional[str]:
        missing = []
        if not self.host:
            missing.append("ADMIN_POSTGRES_HOST (or POSTGRES_HOST)")
        if not self.port:
            missing.append("ADMIN_POSTGRES_PORT (or POSTGRES_PORT)")
        if not self.database:
            missing.append("ADMIN_POSTGRES_DB")
        if not self.user:
            missing.append("ADMIN_POSTGRES_USER (or POSTGRES_USER)")
        if not self.password:
            missing.append("ADMIN_POSTGRES_PASSWORD")

        if missing:
            return f"Missing required admin Postgres env vars: {', '.join(missing)}"

        if not str(self.port).isdigit():
            return "Invalid admin Postgres port (must be numeric)"

        return None


class AdminLogger:
    """
    Logs conversations to admin database for dashboard analytics.
    
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
        self._conn = None
        self.available = POSTGRES_AVAILABLE and (self.config.validate() is None)

        print(f"[AdminLogger] Config: host={self.config.host}, port={self.config.port}, db={self.config.database}, user={self.config.user}")
        print(f"[AdminLogger] POSTGRES_AVAILABLE={POSTGRES_AVAILABLE}, config_valid={self.config.validate() is None}")

        if self.available:
            self._check_connection()
        else:
            # Don't spam errors on import; just mark disabled.
            config_error = self.config.validate()
            reason = config_error or "psycopg2 not installed"
            print(f"[AdminLogger] Disabled ({reason})")
    
    def _check_connection(self):
        """Verify database connection."""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            conn.close()
            print(f"[AdminLogger] Connected to admin database: {self.config.database}")
        except Exception as e:
            print(f"[AdminLogger] Connection failed: {e}")
            self.available = False
    
    def _get_connection(self):
        """Get database connection."""
        if not self.available:
            raise RuntimeError("AdminLogger not available")
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
            sslmode='require'
        )
    
    def _get_or_create_user(self, cur, ms_user_id: str, display_name: str = None, email: str = None) -> int:
        """Get or create user and return user ID."""
        # Try to get existing user
        cur.execute("SELECT id FROM users WHERE ms_user_id = %s", (ms_user_id,))
        row = cur.fetchone()
        
        if row:
            # Update last_active
            cur.execute(
                "UPDATE users SET last_active = NOW(), display_name = COALESCE(%s, display_name), email = COALESCE(%s, email) WHERE id = %s",
                (display_name, email, row[0])
            )
            return row[0]
        
        # Create new user
        cur.execute(
            "INSERT INTO users (ms_user_id, display_name, email) VALUES (%s, %s, %s) RETURNING id",
            (ms_user_id, display_name, email)
        )
        return cur.fetchone()[0]
    
    def _get_or_create_conversation(self, cur, thread_id: str, user_id: int) -> int:
        """Get or create conversation and return conversation ID."""
        # Try to get existing conversation
        cur.execute("SELECT id FROM conversations WHERE thread_id = %s", (thread_id,))
        row = cur.fetchone()
        
        if row:
            # Update last_message_at and increment count
            cur.execute(
                "UPDATE conversations SET last_message_at = NOW(), message_count = message_count + 1 WHERE id = %s",
                (row[0],)
            )
            return row[0]
        
        # Create new conversation
        cur.execute(
            "INSERT INTO conversations (thread_id, user_id, message_count) VALUES (%s, %s, 1) RETURNING id",
            (thread_id, user_id)
        )
        return cur.fetchone()[0]
    
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
        """
        Log a message to the admin database.
        
        Args:
            thread_id: Unique conversation thread ID
            ms_user_id: Microsoft user ID
            role: 'user' or 'assistant'
            content: Message content
            user_name: Display name
            user_email: Email address
            response_time_ms: Response time in milliseconds
            tools_used: List of tools used (for assistant messages)
            sql_query: SQL query executed (if any)
            sql_results_count: Number of SQL results
            error: Error message (if any)
            logs: Detailed generation logs (JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Get or create user
            user_id = self._get_or_create_user(cur, ms_user_id, user_name, user_email)
            
            # Get or create conversation
            conversation_id = self._get_or_create_conversation(cur, thread_id, user_id)
            
            import json
            logs_json = json.dumps(logs) if logs else None
            
            # Insert message
            cur.execute("""
                INSERT INTO messages (
                    conversation_id, role, content, response_time_ms,
                    tools_used, sql_query, sql_results_count, error, logs
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                conversation_id,
                role,
                content,
                response_time_ms,
                tools_used,
                sql_query,
                sql_results_count,
                error,
                logs_json
            ))
            
            conn.commit()
            cur.close()
            conn.close()
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
        """
        Log both user message and assistant response in one call.
        """
        if not self.available:
            return False
        try:
            # Log user message
            self.log_message(
                thread_id=thread_id,
                ms_user_id=ms_user_id,
                role="user",
                content=user_message,
                user_name=user_name,
                user_email=user_email
            )

            # Log assistant response
            self.log_message(
                thread_id=thread_id,
                ms_user_id=ms_user_id,
                role="assistant",
                content=assistant_response,
                user_name=user_name,
                user_email=user_email,
                response_time_ms=response_time_ms,
                tools_used=tools_used,
                sql_query=sql_query,
                sql_results_count=sql_results_count,
                error=error,
                logs=logs
            )

            return True
        except Exception as e:
            print(f"[AdminLogger] Error logging conversation: {e}")
            return False

    def add_feedback(self, ms_user_id: str, feedback: str) -> bool:
        """
        Add feedback to the most recent assistant message for a user.

        Args:
            ms_user_id: Microsoft user ID
            feedback: The feedback text

        Returns:
            True if successful, False otherwise
        """
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Find the most recent assistant message for this user that doesn't have feedback
            cur.execute("""
                UPDATE messages SET feedback = %s, feedback_at = NOW()
                WHERE id = (
                    SELECT m.id FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    JOIN users u ON c.user_id = u.id
                    WHERE u.ms_user_id = %s
                    AND m.role = 'assistant'
                    AND m.feedback IS NULL
                    ORDER BY m.created_at DESC
                    LIMIT 1
                )
                RETURNING id
            """, (feedback, ms_user_id))

            result = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()

            if result:
                print(f"[AdminLogger] Feedback added to message #{result[0]}")
                return True
            else:
                print(f"[AdminLogger] No recent message found for user {ms_user_id[:20]}...")
                return False

        except Exception as e:
            print(f"[AdminLogger] Error adding feedback: {e}")
            return False

    def get_all_conversations(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all conversations with user info for admin dashboard."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                       u.display_name, u.email, u.ms_user_id,
                       (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_message,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) as feedback_count
                FROM conversations c
                JOIN users u ON c.user_id = u.id
                ORDER BY c.last_message_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))

            columns = ['id', 'thread_id', 'message_count', 'created_at', 'last_message_at',
                      'display_name', 'email', 'ms_user_id', 'first_message', 'feedback_count']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting conversations: {e}")
            return []

    def get_conversation_messages(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Get all messages for a specific conversation."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT m.id, m.role, m.content, m.created_at, m.response_time_ms,
                       m.tools_used, m.sql_query, m.sql_results_count, m.error, m.feedback, m.feedback_at
                FROM messages m
                WHERE m.conversation_id = %s
                ORDER BY m.created_at ASC
            """, (conversation_id,))

            columns = ['id', 'role', 'content', 'created_at', 'response_time_ms',
                      'tools_used', 'sql_query', 'sql_results_count', 'error', 'feedback', 'feedback_at']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting messages: {e}")
            return []

    def get_conversation_with_user(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get conversation details with user info."""
        if not self.available:
            return None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                       u.id as user_id, u.display_name, u.email, u.ms_user_id, u.first_seen, u.last_active
                FROM conversations c
                JOIN users u ON c.user_id = u.id
                WHERE c.id = %s
            """, (conversation_id,))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                columns = ['id', 'thread_id', 'message_count', 'created_at', 'last_message_at',
                          'user_id', 'display_name', 'email', 'ms_user_id', 'first_seen', 'last_active']
                return dict(zip(columns, row))
            return None
        except Exception as e:
            print(f"[AdminLogger] Error getting conversation: {e}")
            return None

    def get_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all users for admin dashboard."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT u.id, u.ms_user_id, u.display_name, u.email, u.first_seen, u.last_active,
                       COUNT(DISTINCT c.id) as conversation_count,
                       COUNT(m.id) as message_count
                FROM users u
                LEFT JOIN conversations c ON u.id = c.user_id
                LEFT JOIN messages m ON c.id = m.conversation_id
                GROUP BY u.id
                ORDER BY u.last_active DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))

            columns = ['id', 'ms_user_id', 'display_name', 'email', 'first_seen', 'last_active',
                      'conversation_count', 'message_count']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting users: {e}")
            return []

    def get_all_feedback(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all feedback entries for admin dashboard."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT m.id, m.content as assistant_response, m.feedback, m.feedback_at, m.created_at,
                       u.display_name, u.email,
                       (SELECT content FROM messages WHERE conversation_id = m.conversation_id AND role = 'user'
                        AND created_at < m.created_at ORDER BY created_at DESC LIMIT 1) as user_question
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                JOIN users u ON c.user_id = u.id
                WHERE m.feedback IS NOT NULL
                ORDER BY m.feedback_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))

            columns = ['id', 'assistant_response', 'feedback', 'feedback_at', 'created_at',
                      'display_name', 'email', 'user_question']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting feedback: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        if not self.available:
            return {}
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            stats = {}

            # Total conversations
            cur.execute("SELECT COUNT(*) FROM conversations")
            stats['total_conversations'] = cur.fetchone()[0]

            # Total users
            cur.execute("SELECT COUNT(*) FROM users")
            stats['total_users'] = cur.fetchone()[0]

            # Total messages
            cur.execute("SELECT COUNT(*) FROM messages")
            stats['total_messages'] = cur.fetchone()[0]

            # Messages today
            cur.execute("SELECT COUNT(*) FROM messages WHERE created_at >= CURRENT_DATE")
            stats['messages_today'] = cur.fetchone()[0]

            # Messages this week
            cur.execute("SELECT COUNT(*) FROM messages WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'")
            stats['messages_this_week'] = cur.fetchone()[0]

            # Average response time
            cur.execute("SELECT AVG(response_time_ms) FROM messages WHERE response_time_ms IS NOT NULL")
            avg_time = cur.fetchone()[0]
            stats['avg_response_time_ms'] = round(avg_time) if avg_time else 0

            # Total feedback count
            cur.execute("SELECT COUNT(*) FROM messages WHERE feedback IS NOT NULL")
            stats['total_feedback'] = cur.fetchone()[0]

            # Recent conversations (last 5)
            cur.execute("""
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

            # Recent feedback (last 5)
            cur.execute("""
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

            cur.close()
            conn.close()
            return stats
        except Exception as e:
            print(f"[AdminLogger] Error getting statistics: {e}")
            return {}

    def search_conversations(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search conversations by user name or message content."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            search_pattern = f"%{query}%"
            cur.execute("""
                SELECT DISTINCT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                       u.display_name, u.email
                FROM conversations c
                JOIN users u ON c.user_id = u.id
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE u.display_name ILIKE %s OR u.email ILIKE %s OR m.content ILIKE %s
                ORDER BY c.last_message_at DESC
                LIMIT %s
            """, (search_pattern, search_pattern, search_pattern, limit))

            columns = ['id', 'thread_id', 'message_count', 'created_at', 'last_message_at',
                      'display_name', 'email']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error searching conversations: {e}")
            return []

    def get_user_conversations(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Get all conversations for a specific user."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT c.id, c.thread_id, c.message_count, c.created_at, c.last_message_at,
                       (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY created_at LIMIT 1) as first_message
                FROM conversations c
                WHERE c.user_id = %s
                ORDER BY c.last_message_at DESC
                LIMIT %s
            """, (user_id, limit))

            columns = ['id', 'thread_id', 'message_count', 'created_at', 'last_message_at', 'first_message']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting user conversations: {e}")
            return []

    # ========== DELETE METHODS ==========

    def delete_conversation(self, conversation_id: int) -> bool:
        """Delete a single conversation and all its messages."""
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Delete messages first (foreign key constraint)
            cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))
            # Delete conversation
            cur.execute("DELETE FROM conversations WHERE id = %s RETURNING id", (conversation_id,))
            result = cur.fetchone()

            conn.commit()
            cur.close()
            conn.close()

            if result:
                print(f"[AdminLogger] Deleted conversation #{conversation_id}")
                return True
            return False
        except Exception as e:
            print(f"[AdminLogger] Error deleting conversation: {e}")
            return False

    def delete_user_conversations(self, user_id: int) -> int:
        """Delete all conversations for a user. Returns count of deleted conversations."""
        if not self.available:
            return 0
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Get conversation IDs for this user
            cur.execute("SELECT id FROM conversations WHERE user_id = %s", (user_id,))
            conv_ids = [row[0] for row in cur.fetchall()]

            if not conv_ids:
                cur.close()
                conn.close()
                return 0

            # Delete messages for all these conversations
            cur.execute("DELETE FROM messages WHERE conversation_id = ANY(%s)", (conv_ids,))
            # Delete conversations
            cur.execute("DELETE FROM conversations WHERE user_id = %s", (user_id,))

            conn.commit()
            cur.close()
            conn.close()

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
            # First delete all conversations
            self.delete_user_conversations(user_id)

            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("DELETE FROM users WHERE id = %s RETURNING id", (user_id,))
            result = cur.fetchone()

            conn.commit()
            cur.close()
            conn.close()

            if result:
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
            conn = self._get_connection()
            cur = conn.cursor()

            search_pattern = f"%{query}%"
            cur.execute("""
                SELECT u.id, u.ms_user_id, u.display_name, u.email, u.first_seen, u.last_active,
                       COUNT(DISTINCT c.id) as conversation_count,
                       COUNT(m.id) as message_count
                FROM users u
                LEFT JOIN conversations c ON u.id = c.user_id
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE u.display_name ILIKE %s OR u.email ILIKE %s
                GROUP BY u.id
                ORDER BY u.last_active DESC
                LIMIT %s
            """, (search_pattern, search_pattern, limit))

            columns = ['id', 'ms_user_id', 'display_name', 'email', 'first_seen', 'last_active',
                      'conversation_count', 'message_count']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error searching users: {e}")
            return []

    def search_feedback(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search feedback by user name or feedback text."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            search_pattern = f"%{query}%"
            cur.execute("""
                SELECT m.id, m.content as assistant_response, m.feedback, m.feedback_at, m.created_at,
                       u.display_name, u.email,
                       (SELECT content FROM messages WHERE conversation_id = m.conversation_id AND role = 'user'
                        AND created_at < m.created_at ORDER BY created_at DESC LIMIT 1) as user_question
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                JOIN users u ON c.user_id = u.id
                WHERE m.feedback IS NOT NULL
                AND (u.display_name ILIKE %s OR m.feedback ILIKE %s)
                ORDER BY m.feedback_at DESC
                LIMIT %s
            """, (search_pattern, search_pattern, limit))

            columns = ['id', 'assistant_response', 'feedback', 'feedback_at', 'created_at',
                      'display_name', 'email', 'user_question']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
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
            conn = self._get_connection()
            cur = conn.cursor()

            # Build query dynamically
            conditions = []
            params = []

            if user_id:
                conditions.append("c.user_id = %s")
                params.append(user_id)

            if date_from:
                conditions.append("c.created_at >= %s")
                params.append(date_from)

            if date_to:
                conditions.append("c.created_at <= %s")
                params.append(date_to + " 23:59:59")

            if has_feedback is True:
                conditions.append("(SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) > 0")
            elif has_feedback is False:
                conditions.append("(SELECT COUNT(*) FROM messages WHERE conversation_id = c.id AND feedback IS NOT NULL) = 0")

            if search:
                search_pattern = f"%{search}%"
                conditions.append("(u.display_name ILIKE %s OR EXISTS (SELECT 1 FROM messages WHERE conversation_id = c.id AND content ILIKE %s))")
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
                LIMIT %s OFFSET %s
            """
            params.extend([limit, offset])

            cur.execute(query, params)

            columns = ['id', 'thread_id', 'message_count', 'created_at', 'last_message_at',
                      'display_name', 'email', 'ms_user_id', 'first_message', 'feedback_count']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting filtered conversations: {e}")
            return []

    def get_all_users_simple(self) -> List[Dict[str, Any]]:
        """Get all users (id and name only) for dropdown filters."""
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("SELECT id, display_name FROM users ORDER BY display_name")
            results = [{'id': row[0], 'display_name': row[1] or 'Unbekannt'} for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results
        except Exception as e:
            print(f"[AdminLogger] Error getting users simple: {e}")
            return []

    # ========== LEARNED RULES METHODS ==========

    def get_most_recent_conversation(self, ms_user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent Q&A pair for a user (for rule extraction context).

        Args:
            ms_user_id: Microsoft user ID

        Returns:
            Dict with user_question and assistant_response, or None
        """
        if not self.available:
            return None
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            # Get the most recent assistant message and its preceding user message
            cur.execute("""
                WITH recent_assistant AS (
                    SELECT m.id, m.content as assistant_response, m.conversation_id, m.created_at
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    JOIN users u ON c.user_id = u.id
                    WHERE u.ms_user_id = %s
                    AND m.role = 'assistant'
                    ORDER BY m.created_at DESC
                    LIMIT 1
                )
                SELECT
                    ra.assistant_response,
                    (SELECT content FROM messages
                     WHERE conversation_id = ra.conversation_id
                     AND role = 'user'
                     AND created_at < ra.created_at
                     ORDER BY created_at DESC
                     LIMIT 1) as user_question
                FROM recent_assistant ra
            """, (ms_user_id,))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if row and row[1]:  # Ensure we have both values
                return {
                    'assistant_response': row[0],
                    'user_question': row[1]
                }
            return None

        except Exception as e:
            print(f"[AdminLogger] Error getting recent conversation: {e}")
            return None

    def save_learned_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Save a learned rule to the database.

        Args:
            rule: Dict containing rule_text, category, keywords, source_question, source_feedback, confidence_score

        Returns:
            True if saved successfully
        """
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO learned_rules (
                    rule_text, category, keywords, source_question,
                    source_feedback, confidence_score
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                rule.get('rule_text'),
                rule.get('category'),
                rule.get('keywords', []),
                rule.get('source_question'),
                rule.get('source_feedback'),
                rule.get('confidence_score', 1.0)
            ))

            result = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()

            if result:
                print(f"[AdminLogger] Saved learned rule #{result[0]}: {rule.get('rule_text', '')[:50]}...")
                return True
            return False

        except Exception as e:
            print(f"[AdminLogger] Error saving learned rule: {e}")
            return False

    def get_active_rules(self) -> List[Dict[str, Any]]:
        """
        Get all active learned rules.

        Returns:
            List of rule dicts
        """
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                SELECT id, rule_text, category, keywords, confidence_score, usage_count, created_at
                FROM learned_rules
                WHERE is_active = TRUE
                ORDER BY usage_count DESC, confidence_score DESC, created_at DESC
            """)

            columns = ['id', 'rule_text', 'category', 'keywords', 'confidence_score', 'usage_count', 'created_at']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results

        except Exception as e:
            print(f"[AdminLogger] Error getting active rules: {e}")
            return []

    def increment_rule_usage(self, rule_id: int) -> bool:
        """
        Increment the usage count for a rule.

        Args:
            rule_id: The rule's database ID

        Returns:
            True if updated successfully
        """
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE learned_rules
                SET usage_count = usage_count + 1
                WHERE id = %s
                RETURNING id
            """, (rule_id,))

            result = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()

            return result is not None

        except Exception as e:
            print(f"[AdminLogger] Error incrementing rule usage: {e}")
            return False

    def get_all_rules(self, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """
        Get all learned rules (for admin dashboard).

        Args:
            include_inactive: Whether to include inactive rules

        Returns:
            List of rule dicts
        """
        if not self.available:
            return []
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            if include_inactive:
                cur.execute("""
                    SELECT id, rule_text, category, keywords, source_question, source_feedback,
                           confidence_score, usage_count, is_active, created_at
                    FROM learned_rules
                    ORDER BY created_at DESC
                """)
            else:
                cur.execute("""
                    SELECT id, rule_text, category, keywords, source_question, source_feedback,
                           confidence_score, usage_count, is_active, created_at
                    FROM learned_rules
                    WHERE is_active = TRUE
                    ORDER BY created_at DESC
                """)

            columns = ['id', 'rule_text', 'category', 'keywords', 'source_question', 'source_feedback',
                      'confidence_score', 'usage_count', 'is_active', 'created_at']
            results = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.close()
            conn.close()
            return results

        except Exception as e:
            print(f"[AdminLogger] Error getting all rules: {e}")
            return []

    def toggle_rule_active(self, rule_id: int, is_active: bool) -> bool:
        """
        Activate or deactivate a learned rule.

        Args:
            rule_id: The rule's database ID
            is_active: Whether the rule should be active

        Returns:
            True if updated successfully
        """
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("""
                UPDATE learned_rules
                SET is_active = %s
                WHERE id = %s
                RETURNING id
            """, (is_active, rule_id))

            result = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()

            if result:
                print(f"[AdminLogger] Rule #{rule_id} is_active set to {is_active}")
                return True
            return False

        except Exception as e:
            print(f"[AdminLogger] Error toggling rule active: {e}")
            return False

    def delete_rule(self, rule_id: int) -> bool:
        """
        Delete a learned rule.

        Args:
            rule_id: The rule's database ID

        Returns:
            True if deleted successfully
        """
        if not self.available:
            return False
        try:
            conn = self._get_connection()
            cur = conn.cursor()

            cur.execute("DELETE FROM learned_rules WHERE id = %s RETURNING id", (rule_id,))
            result = cur.fetchone()

            conn.commit()
            cur.close()
            conn.close()

            if result:
                print(f"[AdminLogger] Deleted rule #{rule_id}")
                return True
            return False

        except Exception as e:
            print(f"[AdminLogger] Error deleting rule: {e}")
            return False


# Global instance
admin_logger = AdminLogger()

