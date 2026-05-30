"""
Benchmark and status checks for the Green AI RAG pipeline.
"""

import os
import platform
import sqlite3
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ollama
import sqlite_vec

from config import CHAT_MODEL, DB_PATH, EMBED_MODEL, NUM_CTX, OLLAMA_HOST, TOP_K

_BENCH_QUESTION = "Briefly describe what this document is about in one sentence."


def _ollama_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST)


def check_status() -> dict:
    """Fast health check — no generation."""
    client = _ollama_client()

    # Ollama connectivity + model availability
    models_ok = {"embedding": False, "generation": False}
    ollama_ok = False
    try:
        available = {m["name"] for m in client.list()["models"]}
        ollama_ok = True
        models_ok["embedding"] = EMBED_MODEL in available
        models_ok["generation"] = CHAT_MODEL in available
    except Exception:
        pass

    # DB health
    chunks = 0
    sources = []
    db_ok = False
    try:
        db = sqlite3.connect(DB_PATH)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        chunks = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        sources = [r[0] for r in db.execute("SELECT DISTINCT source FROM chunks").fetchall()]
        db.close()
        db_ok = chunks > 0
    except Exception:
        pass

    return {
        "healthy": ollama_ok and models_ok["embedding"] and models_ok["generation"] and db_ok,
        "ollama": ollama_ok,
        "models": models_ok,
        "database": {"ok": db_ok, "chunks": chunks, "sources": sources},
        "config": {"embed_model": EMBED_MODEL, "chat_model": CHAT_MODEL, "top_k": TOP_K},
    }


def run_benchmark() -> dict:
    client = _ollama_client()

    result = {
        "system": _system_info(),
        "status": check_status(),
        "timings": None,
        "verdict": None,
        "error": None,
    }

    if not result["status"]["healthy"]:
        result["error"] = "System not healthy — check models and database before benchmarking."
        return result

    try:
        result["timings"] = _measure_timings(client)
        result["verdict"] = _make_verdict(result["timings"])
    except Exception as e:
        result["error"] = str(e)

    return result


def _system_info() -> dict:
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "gpu": "unknown",
    }
    # Try to detect GPU via Ollama's running processes endpoint
    try:
        client = _ollama_client()
        ps = client._request("GET", "/api/ps").json()
        models_running = ps.get("models", [])
        if models_running:
            gpu_layers = models_running[0].get("size_vram", 0)
            total_size = models_running[0].get("size", 1)
            pct = int(100 * gpu_layers / total_size) if total_size else 0
            info["gpu"] = f"{pct}% of model on GPU" if pct > 0 else "CPU only"
    except Exception:
        pass
    return info


def _measure_timings(client: ollama.Client) -> dict:
    # 1. Embedding
    t0 = time.perf_counter()
    vec = client.embeddings(model=EMBED_MODEL, prompt=_BENCH_QUESTION)["embedding"]
    embed_ms = int((time.perf_counter() - t0) * 1000)

    # 2. Vector search
    db = sqlite3.connect(DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.row_factory = sqlite3.Row
    packed = struct.pack(f"{len(vec)}f", *vec)

    t0 = time.perf_counter()
    rows = db.execute(
        """
        SELECT c.content, c.source, v.distance
        FROM vec_chunks v JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (packed, TOP_K),
    ).fetchall()
    search_ms = int((time.perf_counter() - t0) * 1000)
    db.close()

    # 3. Generation — use stream=False so Ollama returns timing metadata
    context = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['content']}" for r in rows
    )
    prompt = f"Context:\n{context}\n\nQuestion: {_BENCH_QUESTION}"
    system = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "Keep your answer to one sentence."
    )

    t0 = time.perf_counter()
    resp = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"num_ctx": NUM_CTX},
        stream=False,
    )
    wall_ms = int((time.perf_counter() - t0) * 1000)

    # Extract Ollama's internal timing metadata (nanoseconds)
    prompt_eval_ms = int(resp.get("prompt_eval_duration", 0) / 1e6)
    eval_ms = int(resp.get("eval_duration", 0) / 1e6)
    eval_count = resp.get("eval_count", 0)
    tokens_per_sec = round(eval_count / (eval_ms / 1000), 1) if eval_ms > 0 else 0

    return {
        "embed_ms": embed_ms,
        "search_ms": search_ms,
        "prompt_eval_ms": prompt_eval_ms,   # time to process context (≈ TTFT)
        "generation_ms": eval_ms,
        "tokens_generated": eval_count,
        "tokens_per_sec": tokens_per_sec,
        "total_ms": embed_ms + search_ms + wall_ms,
        "answer_preview": resp["message"]["content"][:120],
    }


def _make_verdict(timings: dict) -> dict:
    tps = timings["tokens_per_sec"]
    ttft = timings["prompt_eval_ms"]

    if tps >= 20:
        rating, colour = "excellent", "#16a34a"
        message = "GPU acceleration is working well."
        recommendation = "You're in good shape. Consider gemma4:e4b for better answer quality."
    elif tps >= 8:
        rating, colour = "good", "#65a30d"
        message = "Decent GPU acceleration detected."
        recommendation = "Performance is usable. An RTX 3060 12GB or better would give a further boost."
    elif tps >= 2:
        rating, colour = "moderate", "#d97706"
        message = "Partial GPU offload or Apple Silicon (non-MLX) detected."
        recommendation = (
            "On Apple Silicon: install Ollama natively and use gemma4:e2b-mlx for ~2× speedup. "
            "On Windows: your GPU VRAM may be too small — the model is spilling to CPU."
        )
    else:
        rating, colour = "poor", "#dc2626"
        message = "CPU-only inference detected."
        recommendation = (
            "Add a GPU with ≥12GB VRAM (e.g. RTX 4070 Super) for ~30–50× faster responses. "
            "On Apple Silicon: install Ollama natively instead of running it in Docker."
        )

    return {
        "rating": rating,
        "colour": colour,
        "message": message,
        "recommendation": recommendation,
        "tokens_per_sec": tps,
        "ttft_ms": ttft,
    }
