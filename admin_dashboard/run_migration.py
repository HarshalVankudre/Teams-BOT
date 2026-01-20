"""Run database schema migration."""
import psycopg2

# Database connection settings
DB_CONFIG = {
    'host': 'rueko-admin-db.cjoa4wkcck71.eu-central-1.rds.amazonaws.com',
    'port': 5432,
    'dbname': 'ruekoadmin',
    'user': 'adminuser',
    'password': 'RuekoAdmin2024!',
    'sslmode': 'require'
}

def run_migration():
    print(f"Connecting to {DB_CONFIG['host']}...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Read and execute schema
    with open('schema.sql', 'r') as f:
        sql = f.read()

    cur.execute(sql)
    conn.commit()

    print("Schema created successfully!")

    # Verify tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = cur.fetchall()
    print(f"Tables created: {[t[0] for t in tables]}")

    cur.close()
    conn.close()

if __name__ == '__main__':
    run_migration()
