"""
Diagnostic script to check admin database connectivity and data.
Run this to identify why the dashboard shows 0 users/chats.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# First check if psycopg2 is available
try:
    import psycopg2
    print("[OK] psycopg2 is installed")
except ImportError:
    print("[ERROR] psycopg2 is NOT installed - run: pip install psycopg2-binary")
    exit(1)

# Check environment variables
print("\n=== Environment Variables ===")
admin_vars = {
    'ADMIN_POSTGRES_HOST': os.getenv('ADMIN_POSTGRES_HOST') or os.getenv('POSTGRES_HOST'),
    'ADMIN_POSTGRES_PORT': os.getenv('ADMIN_POSTGRES_PORT') or os.getenv('POSTGRES_PORT'),
    'ADMIN_POSTGRES_DB': os.getenv('ADMIN_POSTGRES_DB'),
    'ADMIN_POSTGRES_USER': os.getenv('ADMIN_POSTGRES_USER') or os.getenv('POSTGRES_USER'),
    'ADMIN_POSTGRES_PASSWORD': os.getenv('ADMIN_POSTGRES_PASSWORD'),
}

missing = []
for key, value in admin_vars.items():
    if value:
        # Mask password
        display = value[:3] + '***' if 'PASSWORD' in key else value
        print(f"  {key}: {display}")
    else:
        print(f"  {key}: NOT SET")
        missing.append(key)

if missing:
    print(f"\n[ERROR] Missing env vars: {', '.join(missing)}")
    print("The admin logger will be DISABLED without these.")
    exit(1)

# Try to connect
print("\n=== Database Connection ===")
try:
    conn = psycopg2.connect(
        host=admin_vars['ADMIN_POSTGRES_HOST'],
        port=admin_vars['ADMIN_POSTGRES_PORT'],
        dbname=admin_vars['ADMIN_POSTGRES_DB'],
        user=admin_vars['ADMIN_POSTGRES_USER'],
        password=admin_vars['ADMIN_POSTGRES_PASSWORD'],
        sslmode='require'
    )
    print(f"[OK] Connected to {admin_vars['ADMIN_POSTGRES_HOST']}")
except Exception as e:
    print(f"[ERROR] Connection failed: {e}")
    exit(1)

# Check if tables exist
print("\n=== Table Check ===")
cur = conn.cursor()
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
""")
tables = [row[0] for row in cur.fetchall()]
print(f"Tables found: {tables}")

required_tables = ['users', 'conversations', 'messages', 'learned_rules']
missing_tables = [t for t in required_tables if t not in tables]

if missing_tables:
    print(f"\n[ERROR] Missing tables: {missing_tables}")
    print("Run the migration: python admin_dashboard/run_migration.py")
    if 'learned_rules' in missing_tables:
        print("\n[HINT] The 'learned_rules' table is needed for automatic rule extraction.")
        print("This was added recently - you need to re-run the migration to create it.")
    cur.close()
    conn.close()
    exit(1)

print("[OK] All required tables exist")

# Check data counts
print("\n=== Data Counts ===")
cur.execute("SELECT COUNT(*) FROM users")
user_count = cur.fetchone()[0]
print(f"  Users: {user_count}")

cur.execute("SELECT COUNT(*) FROM conversations")
conv_count = cur.fetchone()[0]
print(f"  Conversations: {conv_count}")

cur.execute("SELECT COUNT(*) FROM messages")
msg_count = cur.fetchone()[0]
print(f"  Messages: {msg_count}")

cur.execute("SELECT COUNT(*) FROM learned_rules")
rules_count = cur.fetchone()[0]
print(f"  Learned Rules: {rules_count}")

cur.execute("SELECT COUNT(*) FROM learned_rules WHERE is_active = TRUE")
active_rules_count = cur.fetchone()[0]
print(f"  Active Rules: {active_rules_count}")

cur.execute("SELECT COUNT(*) FROM messages WHERE feedback IS NOT NULL")
feedback_count = cur.fetchone()[0]
print(f"  Messages with Feedback: {feedback_count}")

if user_count == 0 and conv_count == 0:
    print("\n[ISSUE] Database is EMPTY!")
    print("Possible causes:")
    print("  1. Bot is not logging - check ADMIN_POSTGRES_* vars in App Runner")
    print("  2. Bot's admin_logger.available is False - check bot startup logs")
    print("  3. No one has sent messages to the bot yet")
    print("\nTo verify bot logging, check the bot container logs for:")
    print('  "[AdminLogger] Connected to admin database: ruekoadmin"')
    print('  or')
    print('  "[AdminLogger] Disabled (...)"')
else:
    print(f"\n[OK] Database has data: {user_count} users, {conv_count} conversations, {msg_count} messages")
    print("\nIf dashboard still shows 0, check the dashboard's env vars configuration.")

# Show recent data if any
if conv_count > 0:
    print("\n=== Recent Conversations ===")
    cur.execute("""
        SELECT c.id, u.display_name, c.created_at, c.message_count
        FROM conversations c
        JOIN users u ON c.user_id = u.id
        ORDER BY c.created_at DESC LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  #{row[0]}: {row[1]} - {row[2]} ({row[3]} messages)")

# Show learned rules if any
if rules_count > 0:
    print("\n=== Learned Rules ===")
    cur.execute("""
        SELECT id, rule_text, category, is_active, usage_count, created_at
        FROM learned_rules
        ORDER BY created_at DESC LIMIT 10
    """)
    for row in cur.fetchall():
        status = "ACTIVE" if row[3] else "inactive"
        print(f"  #{row[0]} [{status}] ({row[2] or 'no-category'}, used {row[4]}x): {row[1][:60]}...")
        print(f"         Created: {row[5]}")
elif feedback_count > 0:
    print("\n=== Learned Rules Issue ===")
    print(f"[WARN] {feedback_count} messages have feedback but 0 rules extracted!")
    print("Possible causes:")
    print("  1. LLM extraction returned is_actionable=False for all feedback")
    print("  2. Database save failed (check bot logs for [RuleExtraction] messages)")
    print("  3. OpenAI API key issue for gpt-4o-mini calls")
    print("\nRecent feedback entries:")
    cur.execute("""
        SELECT m.feedback, m.feedback_at,
               (SELECT content FROM messages WHERE conversation_id = m.conversation_id
                AND role = 'user' AND created_at < m.created_at
                ORDER BY created_at DESC LIMIT 1) as question
        FROM messages m
        WHERE m.feedback IS NOT NULL
        ORDER BY m.feedback_at DESC LIMIT 5
    """)
    for row in cur.fetchall():
        print(f"  - Feedback: '{row[0][:50]}...' at {row[1]}")
        print(f"    Question: '{(row[2] or 'N/A')[:50]}...'")

cur.close()
conn.close()
print("\n=== Diagnosis Complete ===")
