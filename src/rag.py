"""
RAG query engine: embed query → retrieve chunks → generate answer.
"""

import os
import struct
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
import sqlite_vec

from config import CHAT_MODEL, DB_PATH, EMBED_MODEL, EMBED_DIM, NUM_CTX, PERSONAS, TOP_K

FACTUAL_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the context "
    "passages provided below. If the context does not contain enough information to "
    "answer, say so honestly. Do not make up information."
)

_PERSONAS_BY_ID = {p["id"]: p for p in PERSONAS}


def get_db() -> sqlite3.Connection:
    """Get database connection with sqlite-vec support."""
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    return db


def encode(embedding: list[float]) -> bytes:
    """Encode embedding vector as bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def retrieve(
    db: sqlite3.Connection,
    query: str,
    k: int = TOP_K,
    sources: list[str] | None = None,
) -> list[dict]:
    """Retrieve top-k chunks by vector similarity, optionally filtered by source."""
    vec = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]

    if sources:
        # Filter by sources
        placeholders = ",".join("?" * len(sources))
        rows = db.execute(
            f"""
            SELECT c.source, c.content, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
              AND c.source IN ({placeholders})
            ORDER BY v.distance
            """,
            (encode(vec), k, *sources),
        ).fetchall()
    else:
        # No source filter
        rows = db.execute(
            """
            SELECT c.source, c.content, v.distance
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (encode(vec), k),
        ).fetchall()

    return [dict(r) for r in rows]


def generate(
    query: str,
    context_chunks: list[dict],
    persona_id: str | None = None,
) -> str:
    """Generate answer using LLM with context chunks."""
    if not context_chunks:
        return "I could not find any relevant information in the knowledge base."

    persona = _PERSONAS_BY_ID.get(persona_id) if persona_id else None

    if persona:
        context = "\n\n---\n\n".join(c["content"] for c in context_chunks)
        prompt = f"Memories and background:\n{context}\n\n{query}"
        system = persona["system_prompt"]
    else:
        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['content']}" for c in context_chunks
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        system = FACTUAL_SYSTEM_PROMPT

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"num_ctx": NUM_CTX},
    )
    return resp["message"]["content"]


def query(question: str, persona_id: str | None = None) -> dict:
    """Run a RAG query: retrieve context and generate answer."""
    persona = _PERSONAS_BY_ID.get(persona_id) if persona_id else None
    sources = persona["sources"] if persona else None

    db = get_db()
    try:
        chunks = retrieve(db, question, sources=sources)
        answer = generate(question, chunks, persona_id=persona_id)
    finally:
        db.close()

    return {
        "answer": answer,
        "sources": list({c["source"] for c in chunks}),
        "persona": persona_id,
    }
