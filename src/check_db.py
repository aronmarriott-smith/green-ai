"""
Exits 0 if the vector DB has chunks, 1 if it needs ingestion.
Used by docker-entrypoint.sh and run-native.ps1 / run-native.sh.
"""
import sqlite3
import sys
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "db" / "vectors.db")

if Path(DB_PATH).exists():
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
