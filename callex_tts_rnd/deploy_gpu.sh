#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  CALLEX TTS R&D — GPU Server Deployment Script                  ║
# ║                                                                  ║
# ║  ONE-COMMAND deployment of the Callex TTS engine on your GPU.    ║
# ║  This script:                                                    ║
# ║    1. Clones/updates the repo                                    ║
# ║    2. Creates isolated Python venv                               ║
# ║    3. Installs GPU PyTorch + all dependencies                    ║
# ║    4. Validates the installation                                 ║
# ║    5. Registers with PM2 for auto-restart                        ║
# ║    6. Starts the TTS inference server on port 8124               ║
# ║                                                                  ║
# ║  Usage:                                                          ║
# ║    # First time:                                                 ║
# ║    bash deploy_gpu.sh                                            ║
# ║                                                                  ║
# ║    # Update existing deployment:                                 ║
# ║    bash deploy_gpu.sh --update                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
REPO_URL="https://github.com/daksh-wq/callex-1.git"
INSTALL_DIR="/opt/callex/tts-rnd"
VENV_DIR="$INSTALL_DIR/callex_tts_rnd/.venv"
PM2_NAME="cx-tts-rnd"
TTS_PORT=8124
PYTHON_BIN="python3.10"

# ── Colors ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[$(date +%H:%M:%S)]${NC} $1"; }
ok()   { echo -e "${GREEN}  ✅ $1${NC}"; }
warn() { echo -e "${YELLOW}  ⚠️  $1${NC}"; }
err()  { echo -e "${RED}  ❌ $1${NC}"; }
step() { echo -e "\n${BOLD}━━━ Step $1 ━━━${NC}"; }

# ── Parse Arguments ──────────────────────────────────────────────
UPDATE_ONLY=false
SKIP_DEPS=false
for arg in "$@"; do
    case $arg in
        --update)     UPDATE_ONLY=true ;;
        --skip-deps)  SKIP_DEPS=true ;;
        --port=*)     TTS_PORT="${arg#*=}" ;;
        --dir=*)      INSTALL_DIR="${arg#*=}" ;;
        --help)
            echo "Usage: bash deploy_gpu.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --update       Update existing deployment (git pull + restart)"
            echo "  --skip-deps    Skip dependency installation"
            echo "  --port=PORT    TTS server port (default: 8124)"
            echo "  --dir=DIR      Installation directory (default: /opt/callex/tts-rnd)"
            exit 0
            ;;
    esac
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  🔊 CALLEX TTS R&D — GPU Server Deployment"
echo "═══════════════════════════════════════════════════════════════"
echo "  Install Dir:  $INSTALL_DIR"
echo "  Port:         $TTS_PORT"
echo "  PM2 Name:     $PM2_NAME"
echo "  Update Only:  $UPDATE_ONLY"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
#  STEP 0: Pre-flight Checks
# ═══════════════════════════════════════════════════════════════════
step "0: Pre-flight Checks"

# Check if root or has sudo
if [ "$EUID" -ne 0 ] && ! sudo -n true 2>/dev/null; then
    warn "Not running as root. Some directory creation may fail."
    warn "Run with: sudo bash deploy_gpu.sh"
fi

# Check Python
if command -v $PYTHON_BIN &>/dev/null; then
    ok "Python: $($PYTHON_BIN --version)"
else
    # Fallback to python3
    PYTHON_BIN="python3"
    if command -v $PYTHON_BIN &>/dev/null; then
        ok "Python: $($PYTHON_BIN --version)"
    else
        err "Python 3.10+ not found!"
        exit 1
    fi
fi

# Check NVIDIA GPU
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    ok "GPU: $GPU_NAME ($GPU_MEM MiB)"
else
    err "nvidia-smi not found! Is NVIDIA driver installed?"
    exit 1
fi

# Check CUDA
if command -v nvcc &>/dev/null; then
    ok "CUDA: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    warn "nvcc not found — CUDA toolkit may not be in PATH (PyTorch will use its own)"
fi

# Check git
if command -v git &>/dev/null; then
    ok "Git: $(git --version)"
else
    err "Git not found!"
    exit 1
fi

# Check PM2
if command -v pm2 &>/dev/null; then
    ok "PM2: $(pm2 --version)"
else
    warn "PM2 not found. Install: npm install -g pm2"
    warn "Will run without PM2 (not recommended for production)"
fi

# Check ffmpeg (needed for data pipeline)
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg: $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
else
    warn "ffmpeg not found. Install: apt install ffmpeg"
    warn "Data preparation pipeline won't work without it."
fi

# ═══════════════════════════════════════════════════════════════════
#  STEP 1: Clone / Update Repository
# ═══════════════════════════════════════════════════════════════════
step "1: Repository"

if [ "$UPDATE_ONLY" = true ] && [ -d "$INSTALL_DIR" ]; then
    log "Updating existing repo..."
    cd "$INSTALL_DIR"
    git fetch origin
    git reset --hard origin/main
    ok "Repository updated to latest"
else
    log "Cloning repository..."
    sudo mkdir -p "$(dirname $INSTALL_DIR)"
    sudo chown -R $USER:$USER "$(dirname $INSTALL_DIR)" 2>/dev/null || true
    
    if [ -d "$INSTALL_DIR" ]; then
        warn "Directory exists. Pulling latest..."
        cd "$INSTALL_DIR"
        git pull origin main
    else
        git clone "$REPO_URL" "$INSTALL_DIR"
    fi
    ok "Repository ready at $INSTALL_DIR"
fi

cd "$INSTALL_DIR/callex_tts_rnd"
log "Working directory: $(pwd)"

# ═══════════════════════════════════════════════════════════════════
#  STEP 2: Python Virtual Environment
# ═══════════════════════════════════════════════════════════════════
step "2: Python Environment"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    $PYTHON_BIN -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment exists"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
ok "Activated venv: $(which python3)"

# Upgrade pip
pip install --upgrade pip setuptools wheel -q
ok "pip upgraded"

# ═══════════════════════════════════════════════════════════════════
#  STEP 3: Install Dependencies
# ═══════════════════════════════════════════════════════════════════
step "3: Dependencies"

if [ "$SKIP_DEPS" = false ]; then
    # Install GPU PyTorch first (CUDA 11.8)
    log "Installing GPU PyTorch (CUDA 11.8)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
    ok "PyTorch installed"

    # Install the callex_tts package in editable mode
    log "Installing callex_tts package..."
    pip install -e ".[dev]" -q
    ok "callex_tts package installed"
    
    # Install extra runtime dependencies
    log "Installing extra dependencies..."
    pip install noisereduce faster-whisper requests -q
    ok "Runtime extras installed"
else
    ok "Skipping dependency installation (--skip-deps)"
fi

# ═══════════════════════════════════════════════════════════════════
#  STEP 4: Validate Installation
# ═══════════════════════════════════════════════════════════════════
step "4: Validation"

log "Testing imports..."
python3 -c "
import sys
print(f'  Python: {sys.version}')

import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  VRAM: {mem:.1f} GB')

from callex_tts import __version__
print(f'  callex_tts: v{__version__}')

from callex_tts.text.normalizer import HindiTextNormalizer
norm = HindiTextNormalizer()
result = norm.normalize('₹500 EMI भरना है')
print(f'  Normalizer: ₹500 EMI → {result}')

from callex_tts.text.tokenizer import CallexTokenizer
tok = CallexTokenizer()
print(f'  Tokenizer: {tok.vocab_size} symbols')

from callex_tts.audio.effects import AudioEffectsChain
chain = AudioEffectsChain()
print(f'  Audio Effects: {len(chain.effects)} effects loaded')

from callex_tts.serving.server import app
print(f'  Server: FastAPI app ready')

print()
print('  ✅ All components validated!')
"

if [ $? -ne 0 ]; then
    err "Validation failed! Check errors above."
    exit 1
fi
ok "All imports validated"

# ═══════════════════════════════════════════════════════════════════
#  STEP 5: Configure Environment
# ═══════════════════════════════════════════════════════════════════
step "5: Configuration"

# Create .env file for PM2
cat > .env << EOF
TTS_PORT=$TTS_PORT
PYTHONPATH=$(pwd)/src
NVIDIA_VISIBLE_DEVICES=all
COQUI_TOS_AGREED=1
EOF
ok "Environment file created (.env)"

# Create data directories
mkdir -p data/01_raw_calls data/02_studio_recordings data/04_processed data/05_training_ready
ok "Data directories created"

# Create log directory
mkdir -p logs
ok "Log directory created"

# ═══════════════════════════════════════════════════════════════════
#  STEP 6: PM2 Registration
# ═══════════════════════════════════════════════════════════════════
step "6: PM2 Service Registration"

if command -v pm2 &>/dev/null; then
    # Stop existing if running
    pm2 stop "$PM2_NAME" 2>/dev/null || true
    pm2 delete "$PM2_NAME" 2>/dev/null || true

    # Create PM2 ecosystem config
    cat > ecosystem.config.js << PMEOF
module.exports = {
    apps: [{
        name: "$PM2_NAME",
        script: "-m",
        args: "callex_tts.serving.server",
        interpreter: "$VENV_DIR/bin/python3",
        cwd: "$(pwd)",
        env: {
            TTS_PORT: "$TTS_PORT",
            PYTHONPATH: "$(pwd)/src",
            NVIDIA_VISIBLE_DEVICES: "all",
            COQUI_TOS_AGREED: "1",
        },
        max_memory_restart: "8G",
        restart_delay: 5000,
        max_restarts: 10,
        log_date_format: "YYYY-MM-DD HH:mm:ss",
        error_file: "$(pwd)/logs/tts-error.log",
        out_file: "$(pwd)/logs/tts-out.log",
        merge_logs: true,
    }]
};
PMEOF
    ok "PM2 ecosystem config created"

    # Start with PM2
    log "Starting TTS server with PM2..."
    pm2 start ecosystem.config.js
    pm2 save
    ok "TTS server started as PM2 process '$PM2_NAME'"
else
    warn "PM2 not available. Starting directly..."
    log "Run manually: source $VENV_DIR/bin/activate && make serve"
    
    # Start in background as fallback
    nohup "$VENV_DIR/bin/python3" -m callex_tts.serving.server > logs/tts.log 2>&1 &
    echo $! > logs/tts.pid
    ok "Started in background (PID: $(cat logs/tts.pid))"
fi

# ═══════════════════════════════════════════════════════════════════
#  STEP 7: Health Check
# ═══════════════════════════════════════════════════════════════════
step "7: Health Check"

log "Waiting for server startup..."
sleep 5

for attempt in 1 2 3 4 5; do
    if curl -s "http://localhost:$TTS_PORT/health" > /dev/null 2>&1; then
        HEALTH=$(curl -s "http://localhost:$TTS_PORT/health")
        ok "Server is ONLINE!"
        echo ""
        echo "  Health response:"
        echo "  $HEALTH" | python3 -m json.tool 2>/dev/null || echo "  $HEALTH"
        break
    else
        log "Attempt $attempt/5 — server not ready yet..."
        sleep 5
    fi
done

# ═══════════════════════════════════════════════════════════════════
#  Done!
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✅ DEPLOYMENT COMPLETE!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  📍 Install:    $(pwd)"
echo "  🌐 Server:     http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):$TTS_PORT"
echo "  📊 Health:     curl http://localhost:$TTS_PORT/health"
echo "  📊 Metrics:    curl http://localhost:$TTS_PORT/metrics"
echo "  📖 API Docs:   http://localhost:$TTS_PORT/docs"
echo ""
echo "  ── PM2 Commands ──────────────────────────────────────────"
echo "  pm2 logs $PM2_NAME          # View live logs"
echo "  pm2 restart $PM2_NAME       # Restart server"
echo "  pm2 stop $PM2_NAME          # Stop server"
echo "  pm2 monit                    # Real-time dashboard"
echo ""
echo "  ── Quick Test ────────────────────────────────────────────"
echo "  curl -X POST http://localhost:$TTS_PORT/v2/synthesize \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"text\": \"नमस्ते, कैलेक्स में आपका स्वागत है\"}'"
echo ""
echo "  ── Update Deployment ─────────────────────────────────────"
echo "  bash deploy_gpu.sh --update  # Pull latest + restart"
echo ""
echo "═══════════════════════════════════════════════════════════════"
