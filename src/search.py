"""
Hybrid search: semantic retrieval + BM25 keyword reranking.
No generation — returns ranked passages directly.

Pipeline:
  1. Embed query → retrieve RETRIEVE_K candidates by vector similarity
  2. Score each candidate with BM25 (keyword relevance)
  3. Normalise both scores to [0, 1] within the candidate set
  4. Combine: final = (1 - keyword_weight) * semantic + keyword_weight * keyword
  5. Return top RETURN_N sorted by final score
"""

import os
import re
import sqlite3
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
import sqlite_vec
from rank_bm25 import BM25Okapi

from config import DB_PATH, EMBED_MODEL, EMBED_DIM

RETRIEVE_K     = 15    # candidates pulled from vector DB before reranking
RETURN_N       = 5     # results returned after reranking
KEYWORD_WEIGHT = 0.35  # 0 = pure semantic, 1 = pure keyword


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def search(
    query: str,
    retrieve_k: int = RETRIEVE_K,
    return_n: int = RETURN_N,
    keyword_weight: float = KEYWORD_WEIGHT,
    sources: list[str] | None = None,
) -> list[dict]:
    """Hybrid search: combine semantic + keyword scoring."""
    # ── 1. Semantic retrieval ─────────────────────────────────────────────────
    vec = ollama.embeddings(model=EMBED_MODEL, prompt=query)["embedding"]
    packed = struct.pack(f"{len(vec)}f", *vec)

    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row

    if sources:
        placeholders = ",".join("?" * len(sources))
        rows = db.execute(
            f"""
            SELECT c.source, c.content, v.distance
            FROM vec_chunks v JOIN chunks c ON c.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ? AND c.source IN ({placeholders})
            ORDER BY v.distance
            """,
            (packed, retrieve_k, *sources),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT c.source, c.content, v.distance
            FROM vec_chunks v JOIN chunks c ON c.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (packed, retrieve_k),
        ).fetchall()

    db.close()

    if not rows:
        return []

    chunks = [dict(r) for r in rows]

    # ── 2. BM25 keyword scoring ───────────────────────────────────────────────
    corpus      = [_tokenize(c["content"]) for c in chunks]
    bm25        = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))

    # ── 3. Normalise both to [0, 1] within the candidate set ─────────────────
    distances = [c["distance"] for c in chunks]
    min_d, max_d  = min(distances), max(distances)
    d_range       = max_d - min_d or 1.0

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0

    # ── 4. Combine and rank ───────────────────────────────────────────────────
    results = []
    for chunk, bm25_score, distance in zip(chunks, bm25_scores, distances):
        semantic_sim = (max_d - distance) / d_range   # lower distance → higher sim
        keyword_sim  = float(bm25_score) / max_bm25

        combined = (1.0 - keyword_weight) * semantic_sim + keyword_weight * keyword_sim

        results.append({
            "source":         chunk["source"],
            "content":        chunk["content"],
            "semantic_score": round(semantic_sim, 3),
            "keyword_score":  round(keyword_sim, 3),
            "score":          round(combined, 3),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:return_n]
