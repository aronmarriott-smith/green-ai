"""
Tests for the FastAPI server endpoints (src/server.py).

Run with: pytest

Only happy paths are implemented for now. Other paths worth covering are
left as TODO notes under each endpoint's tests.
"""

from unittest.mock import patch

from src.config import PERSONAS


# ── GET / ────────────────────────────────────────────────────────────────────

def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Green AI" in resp.text


# ── GET /personas ──────────────────────────────────────────────────────────────

def test_personas_returns_configured_personas(client):
    resp = client.get("/personas")
    assert resp.status_code == 200
    assert resp.json() == [
        {"id": p["id"], "name": p["name"], "description": p["description"]}
        for p in PERSONAS
    ]

# TODO: /personas with an empty PERSONAS config -> returns []


# ── POST /query ─────────────────────────────────────────────────────────────────

def test_query_factual_happy_path(client):
    mock_result = {
        "answer": "The story begins on a Wall Street trading floor.",
        "sources": ["american-psycho-script.html"],
        "persona": None,
    }
    with patch("src.server.rag.query", return_value=mock_result) as mock_query:
        resp = client.post("/query", json={"question": "What happens at the start?"})

    assert resp.status_code == 200
    assert resp.json() == mock_result
    mock_query.assert_called_once_with("What happens at the start?", None)


def test_query_persona_happy_path(client):
    persona = PERSONAS[0]
    mock_result = {
        "answer": "Let me tell you about my morning routine...",
        "sources": [persona["sources"][0]],
        "persona": persona["id"],
    }
    with patch("src.server.rag.query", return_value=mock_result) as mock_query:
        resp = client.post(
            "/query",
            json={"question": "Tell me about your morning.", "persona": persona["id"]},
        )

    assert resp.status_code == 200
    assert resp.json() == mock_result
    mock_query.assert_called_once_with("Tell me about your morning.", persona["id"])

# TODO: empty question -> 400 "Question must not be empty."
# TODO: missing "question" key in body -> 400
# TODO: rag.query raises an exception -> 500 with {"detail": str(e)}
# TODO: unknown persona id -> passed through to rag.query, falls back to factual mode
# TODO: malformed JSON body -> 4xx response


# ── POST /search ────────────────────────────────────────────────────────────────

def test_search_happy_path(client):
    mock_results = [
        {
            "source": "american-psycho-script.html",
            "content": "...",
            "semantic_score": 0.91,
            "keyword_score": 0.42,
            "score": 0.74,
        }
    ]
    with patch("src.server.search_module.search", return_value=mock_results) as mock_search:
        resp = client.post("/search", json={"query": "morning routine"})

    assert resp.status_code == 200
    assert resp.json() == {"results": mock_results}
    mock_search.assert_called_once_with("morning routine")

# TODO: empty query -> 400 "Query must not be empty."
# TODO: search raises an exception -> 500 with {"detail": str(e)}
# TODO: no matching results -> 200 with {"results": []}


# ── GET /status ─────────────────────────────────────────────────────────────────

def test_status_happy_path(client):
    mock_status = {
        "healthy": True,
        "ollama": True,
        "models": {"embedding": True, "generation": True},
        "database": {"ok": True, "chunks": 42, "sources": ["american-psycho-script.html"]},
        "config": {"embed_model": "embeddinggemma:300m", "chat_model": "gemma4:e2b", "top_k": 3},
    }
    with patch("src.server.bm.check_status", return_value=mock_status):
        resp = client.get("/status")

    assert resp.status_code == 200
    assert resp.json() == mock_status

# TODO: bm.check_status raises an exception -> 500 with {"detail": str(e)}
# TODO: unhealthy status (Ollama down / empty DB) -> 200 with healthy: false


# ── GET /benchmark ──────────────────────────────────────────────────────────────

def test_benchmark_page_returns_html(client):
    resp = client.get("/benchmark")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Benchmark" in resp.text


# ── POST /benchmark ─────────────────────────────────────────────────────────────

def test_benchmark_run_happy_path(client):
    mock_result = {
        "system": {
            "os": "Windows", "os_version": "10", "cpu": "Intel",
            "python": "3.12.0", "gpu": "100% of model on GPU",
        },
        "status": {"healthy": True},
        "timings": {
            "embed_ms": 12, "search_ms": 3, "load_ms": 0, "prompt_eval_ms": 80,
            "generation_ms": 900, "tokens_generated": 24, "tokens_per_sec": 26.7,
            "total_ms": 995, "answer_preview": "This document is about...",
        },
        "verdict": {
            "rating": "good", "colour": "#65a30d", "message": "GPU acceleration active.",
            "recommendation": "...", "tokens_per_sec": 26.7, "ttft_ms": 80,
        },
        "error": None,
    }
    with patch("src.server.bm.run_benchmark", return_value=mock_result):
        resp = client.post("/benchmark")

    assert resp.status_code == 200
    assert resp.json() == mock_result

# TODO: bm.run_benchmark raises an exception -> 500 with {"detail": str(e)}
# TODO: unhealthy system -> result["error"] set, timings/verdict None, still 200
