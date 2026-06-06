"""
Exits 0 if the vector DB has chunks, 1 if it needs ingestion.
Used by docker-entrypoint.sh and run-native.ps1 / run-native.sh.
"""
import sqlite3
import sys
from pathlib import Path

from config import DB_PATH


def check_db_status() -> int:
    """Check if DB has ingested chunks. Exit 0 if ready, 1 if needs ingestion."""
    if Path(DB_PATH).exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            conn.close()
            if count > 0:
                print(f"  {count} chunks in knowledge base")
                return 0
        except Exception as e:
            print(f"  Warning: DB check failed: {e}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(check_db_status())
