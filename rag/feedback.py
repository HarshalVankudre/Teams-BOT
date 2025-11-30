"""
Feedback Service for Teams Bot
Stores all conversations and user feedback in a separate PostgreSQL database.
"""
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[WARNING] psycopg2 not installed. Feedback storage disabled.")


@dataclass
class FeedbackConfig:
    """Configuration for feedback database (separate from SEMA database)"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: str = os.getenv("POSTGRES_PORT", "5432")
    database: str = "teams_feedback"  # Separate database for feedback
    user: str = os.getenv("POSTGRES_USER", "postgres")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    def to_dict(self) -> Dict[str, str]:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password
        }


class FeedbackService:
    """
    Service for storing conversations and feedback.
    All user messages and AI responses are logged.
    Feedback is optional and linked to the most recent conversation.
    """

    def __init__(self, config: Optional[FeedbackConfig] = None):
        self.config = config or FeedbackConfig()
        self.available = POSTGRES_AVAILABLE and bool(self.config.password)

        if self.available:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM conversations")
                count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                print(f"[Feedback] Connected to {self.config.database}, {count} conversations stored")
            except Exception as e:
                print(f"[Feedback] Connection failed: {e}")
                self.available = False
        else:
            print("[Feedback] Service not available (missing credentials or psycopg2)")

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.config.to_dict())

    def save_conversation(
        self,
        user_id: str,
        user_message: str,
        ai_response: str,
        user_name: Optional[str] = None,
        user_email: Optional[str] = None,
        conversation_thread_id: Optional[str] = None,
        response_time_ms: Optional[int] = None,
        query_type: Optional[str] = None,
        data_source: Optional[str] = None
    ) -> Optional[int]:
        """
        Save a conversation (user message + AI response) to the database.

        Args:
            user_id: Teams user ID
            user_message: The user's message/question
            ai_response: The AI's response
            user_name: Display name of the user
            user_email: Email of the user
            conversation_thread_id: Thread ID for multi-turn conversations
            response_time_ms: Response time in milliseconds
            query_type: Type of query (aggregation, filter, etc.)
            data_source: Data source used (postgres, pinecone, hybrid)

        Returns:
            The ID of the inserted conversation, or None if failed
        """
        if not self.available:
            return None

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            sql = """
                INSERT INTO conversations
                (user_id, user_name, user_email, user_message, ai_response,
                 conversation_thread_id, response_time_ms, query_type, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            cursor.execute(sql, (
                user_id,
                user_name,
                user_email,
                user_message,
                ai_response,
                conversation_thread_id,
                response_time_ms,
                query_type,
                data_source
            ))

            conversation_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

            print(f"[Feedback] Saved conversation #{conversation_id} for user {user_id[:20]}...")
            return conversation_id

        except Exception as e:
            print(f"[Feedback] Error saving conversation: {e}")
            return None

    def add_feedback(
        self,
        user_id: str,
        feedback: str
    ) -> bool:
        """
        Add feedback to the most recent conversation for a user.

        Args:
            user_id: Teams user ID
            feedback: The feedback text

        Returns:
            True if feedback was saved successfully
        """
        if not self.available:
            return False

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Update the most recent conversation for this user that doesn't have feedback
            sql = """
                UPDATE conversations
                SET feedback = %s, feedback_timestamp = %s
                WHERE id = (
                    SELECT id FROM conversations
                    WHERE user_id = %s AND feedback IS NULL
                    ORDER BY created_at DESC
                    LIMIT 1
                )
                RETURNING id, user_message
            """

            cursor.execute(sql, (feedback, datetime.now(), user_id))
            result = cursor.fetchone()

            if result:
                conn.commit()
                print(f"[Feedback] Added feedback to conversation #{result[0]}")
                cursor.close()
                conn.close()
                return True
            else:
                # No conversation without feedback found
                cursor.close()
                conn.close()
                print(f"[Feedback] No recent conversation found for user {user_id[:20]}...")
                return False

        except Exception as e:
            print(f"[Feedback] Error adding feedback: {e}")
            return False

    def get_user_conversations(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversations for a user.

        Args:
            user_id: Teams user ID
            limit: Maximum number of conversations to return

        Returns:
            List of conversation dictionaries
        """
        if not self.available:
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            sql = """
                SELECT id, user_message, ai_response, feedback,
                       created_at, feedback_timestamp, query_type, data_source
                FROM conversations
                WHERE user_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """

            cursor.execute(sql, (user_id, limit))
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()

            return results

        except Exception as e:
            print(f"[Feedback] Error getting conversations: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall feedback statistics"""
        if not self.available:
            return {}

        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            stats = {}

            # Total conversations
            cursor.execute("SELECT COUNT(*) as total FROM conversations")
            stats["total_conversations"] = cursor.fetchone()["total"]

            # Conversations with feedback
            cursor.execute("SELECT COUNT(*) as with_feedback FROM conversations WHERE feedback IS NOT NULL")
            stats["with_feedback"] = cursor.fetchone()["with_feedback"]

            # Unique users
            cursor.execute("SELECT COUNT(DISTINCT user_id) as unique_users FROM conversations")
            stats["unique_users"] = cursor.fetchone()["unique_users"]

            # By query type
            cursor.execute("""
                SELECT query_type, COUNT(*) as count
                FROM conversations
                WHERE query_type IS NOT NULL
                GROUP BY query_type
                ORDER BY count DESC
            """)
            stats["by_query_type"] = [dict(row) for row in cursor.fetchall()]

            # Average response time
            cursor.execute("""
                SELECT AVG(response_time_ms) as avg_response_time
                FROM conversations
                WHERE response_time_ms IS NOT NULL
            """)
            result = cursor.fetchone()
            stats["avg_response_time_ms"] = round(result["avg_response_time"]) if result["avg_response_time"] else None

            cursor.close()
            conn.close()

            return stats

        except Exception as e:
            print(f"[Feedback] Error getting statistics: {e}")
            return {}

    def get_recent_feedback(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent conversations with feedback"""
        if not self.available:
            return []

        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            sql = """
                SELECT id, user_name, user_message, ai_response, feedback,
                       created_at, feedback_timestamp
                FROM conversations
                WHERE feedback IS NOT NULL
                ORDER BY feedback_timestamp DESC
                LIMIT %s
            """

            cursor.execute(sql, (limit,))
            results = [dict(row) for row in cursor.fetchall()]
            cursor.close()
            conn.close()

            return results

        except Exception as e:
            print(f"[Feedback] Error getting recent feedback: {e}")
            return []


# Global instance
feedback_service = FeedbackService()
