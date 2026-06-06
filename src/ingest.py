"""
Ingest HTML files: parse → chunk → embed → store in SQLite-vec.
Usage: python -m src.ingest [--reset]
"""

import argparse
import os
import sqlite3
import struct
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
import sqlite_vec
from bs4 import BeautifulSoup
from tqdm import tqdm

from config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DB_PATH, EMBED_MODEL, EMBED_DIM


def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def init_schema(db: sqlite3.Connection, reset: bool = False) -> None:
    if reset:
        db.execute("DROP TABLE IF EXISTS chunks")
        db.execute("DROP TABLE IF EXISTS vec_chunks")

    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            source  TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL
        )
    """)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[{EMBED_DIM}]
        )
    """)
    db.commit()


def parse_html(path: str) -> str:
    """Parse HTML file and extract text content."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "lxml")

    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    import re
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        # Try to break at a sentence boundary within a tolerance window
        if end < len(text):
            boundary = text.rfind(". ", start, end + 60)
            if boundary != -1 and boundary > start + CHUNK_SIZE // 2:
                end = boundary + 2
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def encode(embedding: list[float]) -> bytes:
    """Encode embedding vector as bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def embed(text: str) -> list[float]:
    """Generate embedding for text using Ollama."""
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


def ingest_file(db: sqlite3.Connection, path: str) -> int:
    """Parse, chunk, embed, and store a single file. Returns number of chunks stored."""
    source = os.path.basename(path)
    print(f"Parsing {source}...")
    text = parse_html(path)
    chunks = chunk_text(text)

    with tqdm(total=len(chunks), desc=source, unit="chunk") as bar:
        for i, chunk in enumerate(chunks):
            vec = embed(chunk)
            cur = db.execute(
                "INSERT INTO chunks (source, chunk_index, content) VALUES (?, ?, ?)",
                (source, i, chunk),
            )
            row_id = cur.lastrowid
            db.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                (row_id, encode(vec)),
            )
            bar.update(1)
            if (i + 1) % 100 == 0:
                db.commit()

    db.commit()
    print(f"Stored {len(chunks)} chunks for {source}")
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Drop and recreate tables")
    args = parser.parse_args()

    db = get_db()
    try:
        init_schema(db, reset=args.reset)

        html_files = list(Path(DATA_DIR).glob("*.html"))
        if not html_files:
            print("No HTML files found in data/ — nothing to ingest.")
            return

        total = 0
        for path in html_files:
            existing = db.execute(
                "SELECT COUNT(*) FROM chunks WHERE source = ?",
                (path.name,),
            ).fetchone()[0]
            if existing and not args.reset:
                print(f"  Skipping {path.name} (already ingested — use --reset to re-ingest)")
                continue
            total += ingest_file(db, str(path))

        print(f"\nIngestion complete. Total chunks in DB: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
