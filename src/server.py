"""
FastAPI server.
  GET  /              — main query UI
  POST /query         — RAG query (factual or persona)
  GET  /personas      — list available personas
  GET  /status        — fast health check (JSON)
  GET  /benchmark     — benchmark UI
  POST /benchmark     — run benchmark (JSON)
Run with: uvicorn src.server:app --reload
"""

import os
import sys

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

sys.path.insert(0, os.path.dirname(__file__))
import benchmark as bm
import rag
import search as search_module
from config import PERSONAS

app = FastAPI(title="Green AI")

# ── Main query UI ─────────────────────────────────────────────────────────────

HTML_FORM = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Green AI</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f4f0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 2rem;
    }
    .card {
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2.5rem;
      width: 100%;
      max-width: 720px;
    }
    h1 { font-size: 1.6rem; color: #1a3d1a; margin-bottom: 0.25rem; }
    .subtitle {
      color: #6b7280; font-size: 0.9rem; margin-bottom: 2rem;
    }
    .subtitle a { color: #16a34a; text-decoration: none; }

    /* Mode selector */
    .mode-row {
      display: flex; gap: 0.5rem; margin-bottom: 1.25rem;
    }
    .mode-btn {
      flex: 1; padding: 0.6rem 0.75rem; border-radius: 8px; cursor: pointer;
      font-size: 0.85rem; font-weight: 600; border: 1.5px solid #d1fae5;
      background: #fff; color: #6b7280; transition: all 0.15s; text-align: center;
    }
    .mode-btn.active {
      background: #f0fdf4; border-color: #16a34a; color: #15803d;
    }
    .mode-btn:hover:not(.active) { border-color: #86efac; color: #374151; }

    /* Persona banner */
    .persona-banner {
      display: none;
      align-items: center; gap: 0.75rem;
      background: #1a1a2e; border-radius: 10px;
      padding: 0.75rem 1rem; margin-bottom: 1.25rem; color: #e2e8f0;
    }
    .persona-banner.active { display: flex; }
    .persona-avatar {
      width: 36px; height: 36px; border-radius: 50%;
      background: #16a34a; display: flex; align-items: center;
      justify-content: center; font-weight: 800; font-size: 1rem;
      color: #fff; flex-shrink: 0;
    }
    .persona-name { font-weight: 700; font-size: 0.95rem; }
    .persona-desc { font-size: 0.78rem; opacity: 0.7; margin-top: 1px; }

    label { display: block; font-weight: 600; color: #374151;
            margin-bottom: 0.5rem; font-size: 0.95rem; }
    textarea {
      width: 100%; border: 1.5px solid #d1fae5; border-radius: 8px;
      padding: 0.75rem 1rem; font-size: 1rem; resize: vertical;
      min-height: 100px; outline: none; transition: border-color 0.2s;
      font-family: inherit;
    }
    textarea:focus { border-color: #22c55e; }
    button#submit-btn {
      margin-top: 1rem; background: #16a34a; color: #fff; border: none;
      border-radius: 8px; padding: 0.75rem 2rem; font-size: 1rem;
      font-weight: 600; cursor: pointer; transition: background 0.2s; width: 100%;
    }
    button#submit-btn:hover:not(:disabled) { background: #15803d; }
    button#submit-btn:disabled { background: #86efac; cursor: not-allowed; }
    button#submit-btn.persona-mode { background: #1a1a2e; }
    button#submit-btn.persona-mode:hover:not(:disabled) { background: #2d2d4e; }

    .spinner { display: none; margin-top: 1.5rem; text-align: center;
               color: #6b7280; font-size: 0.9rem; }
    .spinner.active { display: block; }
    .result { margin-top: 1.5rem; display: none; }
    .result.active { display: block; }
    .result-header {
      display: flex; align-items: center; gap: 0.5rem;
      margin-bottom: 0.5rem;
    }
    .result-label { font-weight: 600; color: #374151; font-size: 0.95rem; }
    .result-persona-tag {
      font-size: 0.72rem; font-weight: 700; background: #1a1a2e;
      color: #86efac; padding: 2px 8px; border-radius: 99px;
    }
    .answer {
      background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px;
      padding: 1rem 1.25rem; color: #14532d; line-height: 1.7;
      white-space: pre-wrap; font-size: 0.95rem;
    }
    .answer.persona-answer {
      background: #0f0f1a; border-color: #2d2d4e; color: #e2e8f0;
      font-style: italic;
    }
    .sources { margin-top: 0.75rem; font-size: 0.8rem; color: #6b7280; }
    .sources span {
      background: #e5e7eb; border-radius: 4px; padding: 2px 8px;
      margin-right: 4px; display: inline-block;
    }
    .error {
      background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
      padding: 1rem 1.25rem; color: #991b1b; font-size: 0.9rem;
    }

    /* Search result cards */
    .search-results { display: flex; flex-direction: column; gap: 0.75rem; }
    .search-card {
      border: 1px solid #e5e7eb; border-radius: 10px;
      padding: 1rem 1.25rem; background: #fafafa;
    }
    .search-card-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 0.6rem;
    }
    .search-card-source {
      font-size: 0.78rem; font-weight: 700; color: #6b7280;
      background: #e5e7eb; padding: 2px 8px; border-radius: 4px;
    }
    .search-card-score {
      font-size: 0.85rem; font-weight: 700; color: #16a34a;
    }
    .search-score-bars { display: flex; flex-direction: column; gap: 4px; margin-bottom: 0.6rem; }
    .score-bar-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.72rem; color: #9ca3af; }
    .score-bar-track {
      flex: 1; height: 4px; background: #e5e7eb; border-radius: 2px; overflow: hidden;
    }
    .score-bar-fill { height: 100%; border-radius: 2px; }
    .bar-semantic { background: #16a34a; }
    .bar-keyword  { background: #2563eb; }
    .search-card-snippet {
      font-size: 0.88rem; color: #374151; line-height: 1.6;
      display: -webkit-box; -webkit-line-clamp: 4;
      -webkit-box-orient: vertical; overflow: hidden;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Green AI</h1>
    <p class="subtitle">
      Ask questions about the ingested documents &nbsp;·&nbsp;
      <a href="/benchmark">⚡ Benchmark</a>
    </p>

    <div class="mode-row" id="mode-row">
      <div class="mode-btn active" data-persona="" data-mode="ask" onclick="selectMode(this)">
        💬 Factual Assistant
      </div>
      <div class="mode-btn" data-persona="" data-mode="search" onclick="selectMode(this)">
        🔍 Search
      </div>
    </div>

    <div class="persona-banner" id="persona-banner">
      <div class="persona-avatar" id="persona-avatar">?</div>
      <div>
        <div class="persona-name" id="persona-name">—</div>
        <div class="persona-desc" id="persona-desc">—</div>
      </div>
    </div>

    <label for="question" id="question-label">Your question</label>
    <textarea id="question" placeholder="e.g. What happens at the beginning of the story?"></textarea>
    <button id="submit-btn" onclick="submitQuery()">Ask</button>

    <div class="spinner" id="spinner">Thinking...</div>

    <div class="result" id="result">
      <div class="result-header">
        <span class="result-label">Answer</span>
        <span class="result-persona-tag" id="persona-tag" style="display:none"></span>
      </div>
      <div class="answer" id="answer-text"></div>
      <div class="sources" id="sources"></div>
    </div>
  </div>

  <script>
    let activePersonaId = '';
    let activeMode = 'ask';   // 'ask' | 'search' | 'persona'
    let personas = [];

    async function loadPersonas() {
      try {
        const r = await fetch('/personas');
        personas = await r.json();
        const row = document.getElementById('mode-row');
        personas.forEach(p => {
          const btn = document.createElement('div');
          btn.className = 'mode-btn';
          btn.dataset.persona = p.id;
          btn.dataset.mode = 'persona';
          btn.dataset.name = p.name;
          btn.dataset.desc = p.description;
          btn.innerHTML = '🎭 ' + p.name;
          btn.onclick = () => selectMode(btn);
          row.appendChild(btn);
        });
      } catch(e) { /* personas unavailable */ }
    }

    function selectMode(btn) {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeMode      = btn.dataset.mode || 'ask';
      activePersonaId = btn.dataset.persona || '';

      const banner    = document.getElementById('persona-banner');
      const label     = document.getElementById('question-label');
      const ta        = document.getElementById('question');
      const submitBtn = document.getElementById('submit-btn');

      // Reset
      banner.classList.remove('active');
      submitBtn.className = '';

      if (activeMode === 'persona') {
        banner.classList.add('active');
        document.getElementById('persona-avatar').textContent = btn.dataset.name[0];
        document.getElementById('persona-name').textContent   = btn.dataset.name;
        document.getElementById('persona-desc').textContent   = btn.dataset.desc;
        label.textContent     = 'Say something to ' + btn.dataset.name;
        ta.placeholder        = 'e.g. Tell me about your morning routine...';
        submitBtn.textContent = 'Send';
        submitBtn.className   = 'persona-mode';
      } else if (activeMode === 'search') {
        label.textContent     = 'Search your documents';
        ta.placeholder        = 'e.g. morning routine business card';
        submitBtn.textContent = 'Search';
      } else {
        label.textContent     = 'Your question';
        ta.placeholder        = 'e.g. What happens at the beginning of the story?';
        submitBtn.textContent = 'Ask';
      }
      document.getElementById('result').classList.remove('active');
    }

    async function submitQuery() {
      const input = document.getElementById('question').value.trim();
      if (!input) return;

      const btn       = document.getElementById('submit-btn');
      const spinner   = document.getElementById('spinner');
      const result    = document.getElementById('result');
      const answerEl  = document.getElementById('answer-text');
      const sourcesEl = document.getElementById('sources');
      const tag       = document.getElementById('persona-tag');

      btn.disabled = true;
      spinner.classList.add('active');
      result.classList.remove('active');

      try {
        if (activeMode === 'search') {
          await runSearch(input, answerEl, sourcesEl, tag);
        } else {
          await runQuery(input, answerEl, sourcesEl, tag);
        }
        result.classList.add('active');
      } catch(e) {
        answerEl.className   = 'error';
        answerEl.textContent = 'Failed to reach the server.';
        result.classList.add('active');
      } finally {
        btn.disabled = false;
        spinner.classList.remove('active');
      }
    }

    async function runQuery(question, answerEl, sourcesEl, tag) {
      answerEl.className = 'answer';
      const body = { question };
      if (activePersonaId) body.persona = activePersonaId;

      const resp = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await resp.json();

      if (!resp.ok) {
        answerEl.className   = 'error';
        answerEl.textContent = data.detail || 'An error occurred.';
        sourcesEl.innerHTML  = '';
        tag.style.display    = 'none';
      } else {
        answerEl.textContent = data.answer;
        if (data.persona) {
          const p = personas.find(x => x.id === data.persona);
          answerEl.className = 'answer persona-answer';
          tag.textContent    = p ? p.name : data.persona;
          tag.style.display  = '';
        } else {
          answerEl.className = 'answer';
          tag.style.display  = 'none';
        }
        sourcesEl.innerHTML = data.sources.length
          ? 'Sources: ' + data.sources.map(s => `<span>${s}</span>`).join('')
          : '';
      }
    }

    async function runSearch(query, answerEl, sourcesEl, tag) {
      tag.style.display   = 'none';
      sourcesEl.innerHTML = '';

      const resp = await fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await resp.json();

      if (!resp.ok || !data.results) {
        answerEl.className   = 'error';
        answerEl.textContent = data.detail || 'Search failed.';
        return;
      }

      if (data.results.length === 0) {
        answerEl.className   = 'answer';
        answerEl.textContent = 'No matching passages found.';
        return;
      }

      const bar = (score, cls) =>
        `<div class="score-bar-track"><div class="score-bar-fill ${cls}" style="width:${Math.round(score*100)}%"></div></div>`;

      answerEl.className = '';
      answerEl.innerHTML = `<div class="search-results">${
        data.results.map((r, i) => `
          <div class="search-card">
            <div class="search-card-header">
              <span class="search-card-source">${r.source}</span>
              <span class="search-card-score">${Math.round(r.score * 100)}% match</span>
            </div>
            <div class="search-score-bars">
              <div class="score-bar-row">
                <span style="width:4.5rem">Semantic</span>
                ${bar(r.semantic_score, 'bar-semantic')}
                <span style="width:2.5rem;text-align:right">${Math.round(r.semantic_score*100)}%</span>
              </div>
              <div class="score-bar-row">
                <span style="width:4.5rem">Keyword</span>
                ${bar(r.keyword_score, 'bar-keyword')}
                <span style="width:2.5rem;text-align:right">${Math.round(r.keyword_score*100)}%</span>
              </div>
            </div>
            <div class="search-card-snippet">${r.content}</div>
          </div>`).join('')
      }</div>`;
    }

    document.getElementById('question').addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submitQuery();
    });

    loadPersonas();
  </script>
</body>
</html>
"""

# ── Benchmark UI ──────────────────────────────────────────────────────────────

BENCHMARK_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Green AI — Benchmark</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #f0f4f0; min-height: 100vh;
      display: flex; align-items: flex-start; justify-content: center; padding: 2rem;
    }
    .card {
      background: #fff; border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 2.5rem; width: 100%; max-width: 720px;
    }
    h1 { font-size: 1.6rem; color: #1a3d1a; margin-bottom: 0.25rem; }
    .subtitle { color: #6b7280; font-size: 0.9rem; margin-bottom: 2rem; }
    .subtitle a { color: #16a34a; text-decoration: none; }
    h2 { font-size: 1rem; font-weight: 700; color: #374151; margin: 1.5rem 0 0.75rem; }
    .row {
      display: flex; justify-content: space-between; align-items: center;
      padding: 0.5rem 0; border-bottom: 1px solid #f3f4f6; font-size: 0.9rem;
    }
    .row:last-child { border-bottom: none; }
    .label { color: #6b7280; }
    .value { font-weight: 600; color: #111827; }
    .badge { display: inline-block; padding: 2px 10px; border-radius: 99px;
             font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
    .ok  { background: #dcfce7; color: #15803d; }
    .err { background: #fee2e2; color: #991b1b; }
    .verdict-box { border-radius: 10px; padding: 1.25rem; margin-top: 1rem; }
    .verdict-rating { font-size: 1.4rem; font-weight: 800; text-transform: uppercase; }
    .verdict-msg { margin-top: 0.4rem; font-size: 0.95rem; }
    .verdict-rec {
      margin-top: 0.75rem; font-size: 0.85rem; opacity: 0.85;
      border-top: 1px solid rgba(0,0,0,0.1); padding-top: 0.75rem;
    }
    button {
      margin-top: 1.5rem; background: #16a34a; color: #fff; border: none;
      border-radius: 8px; padding: 0.75rem 2rem; font-size: 1rem; font-weight: 600;
      cursor: pointer; transition: background 0.2s; width: 100%;
    }
    button:hover:not(:disabled) { background: #15803d; }
    button:disabled { background: #86efac; cursor: not-allowed; }
    .spinner { display: none; margin-top: 1rem; text-align: center;
               color: #6b7280; font-size: 0.9rem; }
    .spinner.active { display: block; }
    #results { display: none; margin-top: 0.5rem; }
    #results.active { display: block; }
    .timing-bar { height: 6px; border-radius: 3px; background: #16a34a;
                  margin-top: 3px; transition: width 0.4s ease; }
  </style>
</head>
<body>
  <div class="card">
    <h1>⚡ Benchmark</h1>
    <p class="subtitle"><a href="/">← Back to Green AI</a></p>

    <div id="status-section">
      <h2>System Status</h2>
      <div id="status-rows"><div class="row"><span class="label">Loading...</span></div></div>
    </div>

    <button id="run-btn" onclick="runBenchmark()">Run Benchmark</button>
    <p style="margin-top:0.5rem;font-size:0.8rem;color:#9ca3af;text-align:center">
      Runs a full RAG query — may take a minute on CPU-only hardware.
    </p>
    <div class="spinner" id="spinner">Running benchmark...</div>

    <div id="results">
      <h2>Timings</h2>
      <div id="timing-rows"></div>
      <div id="verdict-box" class="verdict-box"></div>
    </div>
  </div>

  <script>
    async function loadStatus() {
      try {
        const r = await fetch('/status');
        const d = await r.json();
        const b = ok => `<span class="badge ${ok ? 'ok' : 'err'}">${ok ? 'OK' : 'Error'}</span>`;
        document.getElementById('status-rows').innerHTML = `
          <div class="row"><span class="label">Ollama</span><span class="value">${b(d.ollama)}</span></div>
          <div class="row"><span class="label">Embedding model (${d.config.embed_model})</span><span class="value">${b(d.models.embedding)}</span></div>
          <div class="row"><span class="label">Generation model (${d.config.chat_model})</span><span class="value">${b(d.models.generation)}</span></div>
          <div class="row"><span class="label">Knowledge base</span><span class="value">${d.database.chunks} chunks · ${d.database.sources.join(', ') || 'none'}</span></div>
        `;
      } catch(e) {
        document.getElementById('status-rows').innerHTML =
          '<div class="row"><span class="label" style="color:#dc2626">Could not reach server</span></div>';
      }
    }

    async function runBenchmark() {
      const btn = document.getElementById('run-btn');
      const spinner = document.getElementById('spinner');
      const results = document.getElementById('results');
      btn.disabled = true;
      spinner.classList.add('active');
      results.classList.remove('active');

      try {
        const r = await fetch('/benchmark', { method: 'POST' });
        const d = await r.json();

        if (d.error) {
          document.getElementById('timing-rows').innerHTML =
            `<div class="row"><span class="label" style="color:#dc2626">${d.error}</span></div>`;
          results.classList.add('active');
          return;
        }

        const t = d.timings;
        const fmt = ms => ms >= 1000 ? (ms/1000).toFixed(1)+'s' : ms+'ms';
        const bar = ms => {
          const pct = Math.max(2, Math.round(100 * ms / t.total_ms));
          return `<div class="timing-bar" style="width:${pct}%"></div>`;
        };

        const loadRow = t.load_ms > 500
          ? `<div class="row">
               <span class="label" style="color:#d97706">
                 ⚠ Model cold start
                 <span style="font-size:0.75rem;font-weight:400;display:block;margin-top:1px">
                   Model was evicted from memory. Run again for a warm reading.
                 </span>
               </span>
               <span class="value" style="color:#d97706">${fmt(t.load_ms)}</span>
             </div>${bar(t.load_ms)}`
          : '';

        document.getElementById('timing-rows').innerHTML = `
          <div class="row"><span class="label">Embed query</span><span class="value">${fmt(t.embed_ms)}</span></div>
          ${bar(t.embed_ms)}
          <div class="row"><span class="label">Vector search</span><span class="value">${fmt(t.search_ms)}</span></div>
          ${bar(t.search_ms)}
          ${loadRow}
          <div class="row"><span class="label">Prompt processing</span><span class="value">${fmt(t.prompt_eval_ms)}</span></div>
          ${bar(t.prompt_eval_ms)}
          <div class="row"><span class="label">Generation (${t.tokens_generated} tokens)</span><span class="value">${fmt(t.generation_ms)} · ${t.tokens_per_sec} tok/s</span></div>
          ${bar(t.generation_ms)}
          <div class="row" style="border-top:2px solid #e5e7eb;margin-top:0.5rem;padding-top:0.75rem">
            <span class="label"><strong>Total</strong></span>
            <span class="value"><strong>${fmt(t.total_ms)}</strong></span>
          </div>
          <div class="row" style="font-size:0.8rem;color:#9ca3af">
            <span>Sample answer</span>
            <span style="text-align:right;max-width:65%;font-style:italic">"${t.answer_preview}…"</span>
          </div>
        `;

        const v = d.verdict;
        const vb = document.getElementById('verdict-box');
        vb.style.background = v.colour + '15';
        vb.style.border = '1px solid ' + v.colour + '40';
        vb.innerHTML = `
          <div class="verdict-rating" style="color:${v.colour}">${v.rating} · ${v.tokens_per_sec} tok/s</div>
          <div class="verdict-msg">${v.message}</div>
          <div class="verdict-rec">💡 ${v.recommendation}</div>
        `;

        // Remove stale GPU row from a previous run before adding the fresh one
        document.getElementById('gpu-usage-row')?.remove();
        if (d.system?.gpu && d.system.gpu !== 'unknown') {
          document.getElementById('status-rows').innerHTML +=
            `<div class="row" id="gpu-usage-row"><span class="label">GPU usage</span><span class="value">${d.system.gpu}</span></div>`;
        }

        results.classList.add('active');
      } catch(e) {
        document.getElementById('timing-rows').innerHTML =
          '<div class="row"><span class="label" style="color:#dc2626">Request failed</span></div>';
        results.classList.add('active');
      } finally {
        btn.disabled = false;
        spinner.classList.remove('active');
      }
    }

    loadStatus();
  </script>
</body>
</html>
"""

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_FORM


@app.get("/personas")
async def personas_endpoint():
    return JSONResponse([
        {"id": p["id"], "name": p["name"], "description": p["description"]}
        for p in PERSONAS
    ])


@app.post("/query")
async def query_endpoint(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"detail": "Question must not be empty."}, status_code=400)
    persona_id = body.get("persona") or None
    try:
        result = await run_in_threadpool(rag.query, question, persona_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.post("/search")
async def search_endpoint(request: Request):
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse({"detail": "Query must not be empty."}, status_code=400)
    try:
        results = await run_in_threadpool(search_module.search, query)
        return JSONResponse({"results": results})
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/status")
async def status_endpoint():
    try:
        return JSONResponse(bm.check_status())
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.get("/benchmark", response_class=HTMLResponse)
async def benchmark_page():
    return BENCHMARK_PAGE


@app.post("/benchmark")
async def benchmark_endpoint():
    try:
        result = await run_in_threadpool(bm.run_benchmark)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)
