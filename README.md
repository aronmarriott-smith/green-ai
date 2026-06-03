# Green AI

A local RAG (Retrieval-Augmented Generation) system powered by Google's Gemma models. Ask questions about your documents, search them with hybrid semantic + keyword ranking, or chat with AI personas grounded in them — no data leaves your machine.

This project was built with efficiency as the primary constraint. AI projects of this type are typically expensive to run in production due to the GPU costs involved in cloud inference. This version was rebuilt from the ground up to run on low-powered hardware — including a modern iPhone, or an old MacBook Pro without any GPU access — and then scaled progressively as better hardware becomes available. Features are layered as progressive enhancements: basic search and retrieval works on a minimal CPU instance, and conversational AI or multimodal capabilities are added only when GPU resources justify it.

## How it works

1. **Ingest** — HTML files in `data/` are parsed, split into chunks, and embedded using `embeddinggemma:300m`. Embeddings are stored in a local SQLite vector database.
2. **Query** — Your question is embedded, the most relevant chunks are retrieved and reranked, and `gemma4:e2b` generates an answer grounded in that context.
3. **Search** — Hybrid retrieval combining semantic similarity and BM25 keyword scoring, returning ranked passages with score breakdowns. No generation required.
4. **Persona mode** — Instead of a factual assistant, the model inhabits a character drawn from the documents. Retrieved chunks become the character's memories rather than cited evidence.

## Models

| Role | Model | Size | Licence |
|------|-------|------|---------|
| Embeddings | `embeddinggemma:300m` | 621 MB | Gemma ToU |
| Generation (default) | `gemma4:e2b` | 7.2 GB | Apache 2.0 |
| Generation (Apple Silicon) | `gemma4:e2b-mlx` | 7.1 GB | Apache 2.0 |
| Generation (NVIDIA) | `gemma4:e2b-nvfp4` | 7.1 GB | Apache 2.0 |

A GPU with ≥12 GB VRAM is strongly recommended for response times under 10 seconds.

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

Use this path on M-series Macs to get Metal/MLX acceleration. Docker on macOS cannot pass the GPU through to containers, so native Ollama is required for hardware acceleration.

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
| Intel MBP — i7, Iris 655 | Docker / Native — CPU only | ~90s |
| MBP M4 Pro — 14-core, Metal 4 | Native (`run-native.sh`) + MLX | ~4–8s |
| Mac Mini M1 | Native (`run-native.sh`) + MLX | ~15–25s |
| Windows + GTX 1060 6GB | Docker (partial GPU offload) | ~40–60s |
| Windows + RTX 4070 Super 12GB | Docker (full VRAM) | ~5–10s |

**Why the M4 Pro is faster than the M1:** The M4 Pro's unified memory bandwidth is ~273 GB/s vs ~68 GB/s on the M1. LLM inference on CPU/Apple Silicon is memory-bandwidth bound, so this directly translates to ~4× faster token generation. Combined with MLX optimisations and Metal 4 compute, the M4 Pro is broadly comparable to a mid-range NVIDIA GPU for this workload.

**VRAM rule of thumb:** Both models together (7.2 GB generation + 0.6 GB embedding) need ~8 GB VRAM to run fully on GPU. Any card with ≥12 GB avoids partial CPU offload.

---

## Moving to production

### Design philosophy — progressive enhancement

The application is deliberately tiered so you only pay for what you need:

| Tier | Features available | Minimum hardware |
|------|--------------------|-----------------|
| **1 — Search** | Embedding, hybrid search, BM25 reranking | CPU instance (t3.small) |
| **2 — Conversational AI** | RAG Q&A, persona mode | GPU instance (16 GB VRAM) |
| **3 — Multimodal** | Image search, vision queries | GPU instance (24 GB VRAM) |

Start at Tier 1, validate the use case, then move up only when the value justifies the cost.

---

### Cloud hardware

> **Note on current platform:** Our existing cloud platform may not be able to provision GPU instances — this needs confirmation before committing to an AWS-based approach.

**Ollama requires an NVIDIA GPU.** AMD (ROCm) is supported but less stable. Apple Silicon is excellent for local use but not available in standard cloud instances.

**Tier 1 — CPU only (search and retrieval only)**

| Instance | RAM | vCPU | Approx. cost | Notes |
|----------|-----|------|-------------|-------|
| AWS t3.small | 2 GB | 2 | ~£13/month | Minimum — `embeddinggemma:300m` fits, search endpoint only |

Replacing Ollama with `fastembed` for the embedding step reduces memory below 512 MB, making a t3.micro viable (~£9/month).

**Tier 2 and 3 — GPU required**

16 GB VRAM is sufficient for Tier 2. Both models (`gemma4:e2b` at 7.2 GB + `embeddinggemma:300m` at 0.6 GB) fit with headroom. 24 GB provides room to grow into larger or multimodal models.

| Instance | GPU | VRAM | Tier | Approx. cost | Notes |
|----------|-----|------|------|-------------|-------|
| **AWS g4dn.xlarge** | NVIDIA T4 | 16 GB | 2 | **~£350/month** | Minimum viable for conversational AI |
| **AWS g5.xlarge** | NVIDIA A10G | 24 GB | 2 + 3 | ~£580/month | Recommended — headroom for growth and multimodal |
| AWS p4d / p5 | A100 / H100 | 40–80 GB | — | £3,000+/month | Enterprise — significantly over-specified for this application |

**Recommended path:** Start with g4dn.xlarge for Tier 2. Move to g5.xlarge only if you need multimodal (image search) or find the T4's 16 GB VRAM limiting as the model library grows. The A100/H100 tier is not appropriate for this application and is unlikely to ever be needed at this scale.

**Lower-cost alternatives (if AWS is not required):**

| Provider | GPU | VRAM | Approx. cost |
|----------|-----|------|-------------|
| Vast.ai / RunPod | RTX 3090 | 24 GB | ~£0.25–0.40/hr |
| Lambda Labs | A10 | 24 GB | ~£0.45/hr |

These are suitable for development and staging; for production uptime commitments AWS reserved instances reduce the monthly cost significantly versus on-demand pricing.

---

### Software modifications for production

**1. Multi-Token Prediction (MTP)**
Speculative decoding using a paired drafter model. Google claims up to **3× faster generation** with no quality loss.
- Requires HuggingFace Transformers — not currently compatible with Ollama
- Target model: `google/gemma-4-E2B-it`, Drafter: `google/gemma-4-E2B-it-assistant`
- Most effective on GPU at higher batch sizes; gains at batch size 1 (single-user) are lower due to Gemma 4's MoE architecture
- Best pursued once migrated off Ollama for a high-throughput production deployment

**2. Replace Ollama with vLLM or SGLang**
Both are production inference servers that support continuous batching (handling multiple concurrent users efficiently) and PagedAttention (more efficient VRAM use). Both support Gemma models and enable MTP.
- Ollama is excellent for local/single-user use; under concurrent load a dedicated inference server becomes worthwhile
- Migration cost: moderate — the inference API is compatible but model loading and configuration differ

**3. Hardware-specific quantization**
Switch the generation model variant to match the deployment hardware:
- NVIDIA GPU → `gemma4:e2b-nvfp4` (~10–20% faster than Q4_K_M via native CUDA FP4)
- Apple Silicon → `gemma4:e2b-mlx` (MLX-optimised, already used by `run-native.sh`)
- CPU-only → current `gemma4:e2b` (Q4_K_M) is already optimal

**4. Database: sqlite-vec → pgvector**
sqlite-vec works well for single-user and low-concurrency use. Under production load (multiple simultaneous queries), replace with [pgvector](https://github.com/pgvector/pgvector) on PostgreSQL for concurrent read access and connection pooling. Minimal code change — the query interface is similar.

**5. Content safety layer**
Add `gemma3:270m` as a guard model checking both user input and model output before any response is returned. Fast (~750ms overhead per query), same model family as the generation model. Essential before any public-facing deployment.

**6. Authentication and rate limiting**
Add API key authentication and per-IP rate limiting to all POST endpoints using FastAPI middleware (`slowapi`). The `/benchmark` endpoint should be restricted to internal access only.

---

## Persona mode

Personas are configured in `src/config.py`. Each persona has a name, a system prompt that describes the character's voice, and a list of source documents whose chunks are used as the character's memories.

The UI displays a mode selector — switch between **Factual Assistant**, **Search**, and any configured persona before submitting.

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
│   ├── search.py           # Hybrid search: semantic + BM25 reranking
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

Key settings in `src/config.py`. Model names can be overridden with environment variables:

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
| `/search` | POST | Hybrid search — `{"query": "..."}` — no generation |
| `/personas` | GET | List available personas |
| `/status` | GET | Health check (JSON) |
| `/benchmark` | GET | Benchmark UI |
| `/benchmark` | POST | Run full benchmark (JSON) |

## Planned features

**Core**
- PDF ingestion support
- Conversational chat history

**Safety and access control**
- Content safety layer — `gemma3:270m` as input/output guard model for public deployments
- Authentication and rate limiting

**Performance**
- Multi-Token Prediction (MTP) — up to 3× generation speedup via speculative decoding; requires migration from Ollama to HuggingFace Transformers or vLLM, and a GPU
- Lightweight cloud search endpoint — replace Ollama embedding with `fastembed` to run Tier 1 search-only on a t3.micro (~£9/month)
- pgvector migration for production concurrent access

**Multimodal (Tier 3 — requires 24 GB VRAM)**
- Image search and vision queries — `gemma4:e2b` already supports image input; requires multimodal ingestion pipeline and a GPU instance with sufficient VRAM (g5.xlarge recommended)
