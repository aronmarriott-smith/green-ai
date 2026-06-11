#Requires -Version 5.1
<#
.SYNOPSIS
    Native deployment script for Windows -- PowerShell equivalent of run-native.sh.
    Recommended for: Windows with NVIDIA GPU (without Docker).

.DESCRIPTION
    - Detects NVIDIA GPU via nvidia-smi and selects gemma4:e2b-nvfp4
    - Checks Ollama is installed, starts it if not already running
    - Pulls embedding and chat models on first run
    - Creates a Python 3.10+ virtual environment and installs dependencies
    - Runs ingestion if the knowledge base is empty
    - Starts the FastAPI server at http://localhost:8000

.EXAMPLE
    # Allow script execution (one-time per user, if not already set):
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

    .\run-native.ps1

    # Override models via environment variables before running:
    $env:CHAT_MODEL = "gemma4:e2b"; .\run-native.ps1
#>

$ErrorActionPreference  = "Stop"
$ProgressPreference     = "SilentlyContinue"   # suppress Invoke-WebRequest progress bars

# -- Detect hardware and set model variant ------------------------------------
$Platform   = "Windows"
$ChatModel  = if ($env:CHAT_MODEL)  { $env:CHAT_MODEL }  else { "gemma4:e2b" }
$EmbedModel = if ($env:EMBED_MODEL) { $env:EMBED_MODEL } else { "embeddinggemma:300m" }

if (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name --format=csv,noheader | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $Platform = "Windows (NVIDIA GPU)"
        # Use the cross-platform Gemma model on Windows; Ollama uses CUDA behind the scenes.
        if (-not $env:CHAT_MODEL) {
            $ChatModel = "gemma4:e2b"
        }
    }
}

$env:CHAT_MODEL  = $ChatModel
$env:EMBED_MODEL = $EmbedModel

Write-Host ""
Write-Host "  Green AI - Native"
Write-Host "  ================="
Write-Host "  Platform:    $Platform"
Write-Host "  Chat model:  $ChatModel"
Write-Host "  Embed model: $EmbedModel"
Write-Host ""

# -- Check Ollama is installed ------------------------------------------------
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Host ""
    Write-Host "  Ollama is not installed."
    Write-Host ""
    Write-Host "  1. Download and install it from: https://ollama.com"
    Write-Host "  2. Re-run this script."
    Write-Host ""
    exit 1
}

# -- Ollama readiness check ---------------------------------------------------
function Test-OllamaReady {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" `
            -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        return ($r.StatusCode -eq 200)
    } catch {
        return $false
    }
}

# -- Start Ollama if not already running --------------------------------------
if (Test-OllamaReady) {
    Write-Host "  Ollama already running"
} else {
    Write-Host "  Starting Ollama..."
    $env:OLLAMA_KEEP_ALIVE = "-1"
    if ($Platform -eq "Windows (NVIDIA GPU)") {
        $env:OLLAMA_NUM_GPU = "99"
    } else {
        $env:OLLAMA_NUM_GPU = "0"
    }
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Write-Host "  Waiting for Ollama to be ready..."
    $i = 0
    while (-not (Test-OllamaReady)) {
        if ($i -ge 30) {
            Write-Host "  Error: Ollama did not start within 60 seconds."
            exit 1
        }
        Start-Sleep -Seconds 2
        $i++
    }
    Write-Host "  Ollama is ready"
}

# -- Pull models if not already downloaded ------------------------------------
function Invoke-PullIfMissing ([string]$Model) {
    $body = "{`"name`":`"$Model`"}"
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/show" `
            -Method POST -Body $body -ContentType "application/json" `
            -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        Write-Host "  [cached] $Model"
    } catch {
        Write-Host "  Downloading $Model - first run only, may take several minutes..."
        ollama pull $Model
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  Error: Failed to pull $Model"
            exit 1
        }
        Write-Host "  [ready]  $Model"
    }
}

Write-Host "  Checking models..."
Invoke-PullIfMissing $EmbedModel
Invoke-PullIfMissing $ChatModel

# -- Find Python 3.10+ --------------------------------------------------------
$PythonExe = $null
foreach ($candidate in @("python", "python3", "py")) {
    if (-not (Get-Command $candidate -ErrorAction SilentlyContinue)) { continue }
    try {
        $ver = (& $candidate --version 2>&1)
        if ("$ver" -match "Python (\d+)\.(\d+)") {
            if ([int]$Matches[1] -ge 3 -and [int]$Matches[2] -ge 10) {
                $PythonExe = $candidate
                break
            }
        }
    } catch {}
}

if (-not $PythonExe) {
    Write-Host ""
    Write-Host "  Python 3.10 or newer is required but was not found."
    Write-Host ""

    $hasWinget = [bool](Get-Command "winget" -ErrorAction SilentlyContinue)
    if ($hasWinget) {
        Write-Host "  [1] Install Python automatically via winget"
        Write-Host "  [2] I will install it myself"
    } else {
        Write-Host "  [1] I will install it myself"
    }
    Write-Host ""
    $choice = Read-Host "  Choose an option"

    if ($hasWinget -and $choice -eq "1") {
        Write-Host ""
        Write-Host "  Installing Python 3.12..."
        winget install --id Python.Python.3.12 --source winget
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "  Installation failed. Please install manually:"
            Write-Host "  https://www.python.org/downloads/"
            Write-Host "  Check 'Add Python to PATH' during setup, then re-run this script."
        } else {
            Write-Host ""
            Write-Host "  Python installed. Open a new terminal window and re-run this script."
        }
    } else {
        Write-Host ""
        Write-Host "  Install Python 3.10+ from: https://www.python.org/downloads/"
        Write-Host "  Check 'Add Python to PATH' during setup, then re-run this script."
    }
    exit 0
}

Write-Host "  Using Python: $((& $PythonExe --version 2>&1))"

# -- Virtual environment paths ------------------------------------------------
$VenvPython  = "venv\Scripts\python.exe"
$VenvPip     = "venv\Scripts\pip.exe"
$VenvUvicorn = "venv\Scripts\uvicorn.exe"

# -- Recreate venv if SQLite extension support is missing (needed by sqlite-vec)
if (Test-Path "venv") {
    & $VenvPython -c "import sqlite3; sqlite3.connect(':memory:').enable_load_extension(True)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Recreating venv - SQLite extension support missing..."
        Remove-Item -Recurse -Force "venv"
    }
}

if (-not (Test-Path "venv")) {
    Write-Host "  Creating Python virtual environment..."
    & $PythonExe -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Error: Failed to create virtual environment."
        exit 1
    }
}

# -- Install dependencies if not already present ------------------------------
& $VenvPip show fastapi | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing Python dependencies..."
    & $VenvPip install -q -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Error: Failed to install dependencies."
        exit 1
    }
}

# -- Run ingestion if the knowledge base is empty -----------------------------
Write-Host ""
Write-Host "  Checking knowledge base..."
& $VenvPython src\check_db.py | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Running ingestion - this may take a while on first run..."
    & $VenvPython -u src\ingest.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Error: Ingestion failed."
        exit 1
    }
} else {
    Write-Host "  Knowledge base OK"
}

# -- Start the server ---------------------------------------------------------
Write-Host ""
Write-Host "  Server ready -> http://localhost:8000"
Write-Host "  Press Ctrl+C to stop."
Write-Host ""
& $VenvUvicorn src.server:app --host 0.0.0.0 --port 8000
