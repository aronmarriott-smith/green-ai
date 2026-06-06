#!/bin/bash
set -e

echo ""
echo "  Green AI"
echo "  ========"

# ── 1. Wait for Ollama ──────────────────────────────────────────────────────
echo ""
echo "  Waiting for Ollama at ${OLLAMA_HOST}..."
until curl -sf "${OLLAMA_HOST}/api/tags" > /dev/null 2>&1; do
    sleep 3
done
echo "  Ollama is ready"

# ── 2. Pull models if not already downloaded ────────────────────────────────
pull_model() {
    local model=$1
    # /api/show returns non-zero if the model isn't present locally
    if curl -sf -X POST "${OLLAMA_HOST}/api/show" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"${model}\"}" > /dev/null 2>&1; then
        echo "  [cached] ${model}"
    else
        echo "  Downloading ${model} — first run only, may take several minutes..."
        python3 -c "
import os, ollama
client = ollama.Client(host=os.environ['OLLAMA_HOST'])
last = ''
for chunk in client.pull('${model}', stream=True):
    s = chunk.get('status', '')
    if s and s != last:
        print(f'    {s}', flush=True)
        last = s
"
        echo "  [ready]  ${model}"
    fi
}

echo ""
echo "  Checking models..."
pull_model "embeddinggemma:300m"
pull_model "gemma2:27b-instruct-q4_K_M"

# ── 3. Run ingestion if the knowledge base is empty ─────────────────────────
echo ""
echo "  Checking knowledge base..."

python3 /app/src/check_db.py
INGEST_STATUS=$?

if [ $INGEST_STATUS -eq 0 ]; then
    echo "  Knowledge base OK"
elif [ $INGEST_STATUS -eq 1 ]; then
    echo "  Running ingestion (first run — may take ~15 minutes)..."
    python3 -u /app/src/ingest.py
    echo "  Ingestion complete"
fi

# ── 4. Start the server ─────────────────────────────────────────────────────
echo ""
echo "  Server ready → http://localhost:8000"
echo ""
exec uvicorn src.server:app --host 0.0.0.0 --port 8000
