# Green AI

A local RAG (Retrieval-Augmented Generation) system powered by Google's Gemma models, running entirely on your own hardware via Ollama. Ask questions about your documents, or chat with AI personas grounded in them — no data leaves your machine.

## How it works

1. **Ingest** — HTML files in `data/` are parsed, split into chunks, and embedded using `embeddinggemma:300m`. Embeddings are stored in a local SQLite vector database.
2. **Query** — Your question is embedded, the most relevant chunks are retrieved, and `gemma4:e2b` generates an answer grounded in that context.
3. **Persona mode** — Instead of a factual assistant, the model inhabits a character drawn from the documents. Retrieved chunks become the character's memories rather than cited evidence.

## Models

| Role | Model | Size |
|------|-------|------|
| Embeddings | `embeddinggemma:300m` | 621 MB |
| Generation (default) | `gemma4:e2b` | 7.2 GB |
| Generation (Apple Silicon) | `gemma4:e2b-mlx` | 7.1 GB |
| Generation (NVIDIA) | `gemma4:e2b-nvfp4` | 7.1 GB |

Both are Apache 2.0 licensed and run locally. A GPU with ≥12 GB VRAM is strongly recommended for response times under 10 seconds.

---

## Deployment

### Option A — Docker (recommended for NVIDIA GPU or CPU-only)

**Prerequisites:** [Docker](https://docs.docker.com/get-docker/) with Compose. For GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

1. Add your HTML files to `data/`.

2. Edit `docker-compose.yml` for your hardware:
   - **NVIDIA GPU** — leave the `deploy:` block as-is
   - **CPU only** — remove or comment out the `deploy:` block
   - **AMD GPU** — change the image to `ollama/ollama:rocm` and remove `deploy:`

3. Start:
   ```bash
   docker compose up --build
   ```

On first run the container downloads models and ingests your data automatically (~15 min for large documents). Subsequent starts are near-instant — models and the DB are persisted in Docker volumes.

**Re-ingest after adding new files:**
```bash
docker compose exec app python3 src/ingest.py
# or wipe and re-embed everything:
docker compose exec app python3 src/ingest.py --reset
```

---

### Option B — Native (recommended for Apple Silicon)

Use this path on the M1/M2/M3 Mac Mini to get Metal/MLX acceleration. Docker on macOS cannot pass the GPU through to containers, so native Ollama is required for hardware acceleration.

```bash
chmod +x run-native.sh
./run-native.sh
```

The script automatically:
- Detects Apple Silicon and selects `gemma4:e2b-mlx`
- Starts Ollama natively with `KEEP_ALIVE=-1` (model stays warm)
- Pulls models on first run
- Runs ingestion if the database is empty
- Starts the server at `http://localhost:8000`

On subsequent runs it skips everything already done and starts in seconds.

**Keep Ollama warm between restarts** — add to `~/.zshrc`:
```bash
export OLLAMA_KEEP_ALIVE=-1
```

---

### Option C — Native (Windows with NVIDIA GPU, without Docker)

```bash
# Install Ollama from https://ollama.com, then:
./run-native.sh          # Git Bash / WSL
# or on PowerShell:
python -m venv venv
venv\Scripts\pip install -r requirements.txt
ollama pull embeddinggemma:300m
ollama pull gemma4:e2b
python src\ingest.py
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

For NVIDIA on Windows, Docker + nvidia-container-toolkit (Option A) is the cleaner path.

---

## Hardware guide

| Machine | Recommended path | Expected query time |
|---------|-----------------|-------------------|
| Intel MBP (this dev machine) | Docker / Native — CPU only | ~90s |
| Mac Mini M1 | Native (`run-native.sh`) | ~15–25s |
| Windows + GTX 1060 6GB | Docker (partial GPU offload) | ~40–60s |
| Windows + RTX 4070 Super 12GB | Docker (full VRAM) | ~5–10s |

---

## Persona mode

Personas are configured in `src/config.py`. Each persona has a name, a system prompt that describes the character's voice, and a list of source documents whose chunks are used as the character's memories.

The UI displays a mode selector — switch between **Factual Assistant** and any configured persona before submitting a question.

**Adding a new persona** — add an entry to `PERSONAS` in `src/config.py`:
```python
{
    "id": "my-persona",
    "name": "Character Name",
    "description": "Short description shown in the UI",
    "sources": ["my-document.html"],   # must be ingested first
    "system_prompt": "You are ...",
}
```

---

## Project structure

```
green-ai/
├── data/                   # Drop HTML files here for ingestion
├── db/                     # SQLite vector database (auto-created)
├── src/
│   ├── config.py           # Models, paths, chunking settings, personas
│   ├── ingest.py           # Parse → chunk → embed → store
│   ├── rag.py              # Embed query → retrieve → generate
│   ├── server.py           # FastAPI server + HTML UI
│   ├── benchmark.py        # Performance benchmark logic
│   └── check_db.py         # DB health check (used by entrypoints)
├── Dockerfile
├── docker-compose.yml      # Docker deployment (NVIDIA GPU / CPU)
├── docker-entrypoint.sh    # Container startup: models, ingest, server
├── run-native.sh           # Native deployment (Apple Silicon / bare metal)
└── requirements.txt
```

## Configuration

Key settings in `src/config.py`. Model names can be overridden with environment variables — useful for switching to hardware-optimised variants without editing code:

| Setting | Default | Env var override |
|---------|---------|-----------------|
| `EMBED_MODEL` | `embeddinggemma:300m` | `EMBED_MODEL=...` |
| `CHAT_MODEL` | `gemma4:e2b` | `CHAT_MODEL=gemma4:e2b-mlx` |
| `OLLAMA_HOST` | `http://localhost:11434` | `OLLAMA_HOST=...` |
| `TOP_K` | `3` | — |
| `NUM_CTX` | `2048` | — |
| `CHUNK_SIZE` | `500` | — |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main query UI |
| `/query` | POST | RAG query — `{"question": "...", "persona": "id"}` |
| `/personas` | GET | List available personas |
| `/status` | GET | Health check (JSON) |
| `/benchmark` | GET | Benchmark UI |
| `/benchmark` | POST | Run benchmark (JSON) |

## Planned features

- PDF support
- Conversational chat history
