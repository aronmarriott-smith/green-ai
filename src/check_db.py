"""
Exits 0 if the vector DB has chunks, 1 if it needs ingestion.
Used by docker-entrypoint.sh.
"""
import os
import sqlite3
import sys

DB_PATH = "/app/db/vectors.db"

if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        if count > 0:
            print(f"  {count} chunks in knowledge base")
            sys.exit(0)
    except Exception:
        pass

sys.exit(1)
