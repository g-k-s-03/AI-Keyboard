#!/usr/bin/env bash
# Self-contained benchmark runner for Termux (Android), VirtualBox/Linux VMs,
# and bare Linux. Detects the environment, installs what's missing, sets up
# a venv, installs slm_eval, and runs the mobile-friendly benchmark.
#
# Usage: ./termux_benchmark.sh [model] [category] [timeout_per_prompt]
#   ./termux_benchmark.sh
#   ./termux_benchmark.sh Qwen/Qwen2.5-0.5B-Instruct english_grammar 60

set -euo pipefail

# ---------- environment detection ----------
detect_env() {
    if [ -d "/data/data/com.termux" ] || [ -n "${TERMUX_VERSION:-}" ]; then
        echo "termux"
    elif grep -qi "virtualbox" /sys/class/dmi/id/product_name 2>/dev/null \
        || grep -qi "virtualbox" /sys/class/dmi/id/sys_vendor 2>/dev/null; then
        echo "virtualbox"
    elif grep -qi "microsoft" /proc/version 2>/dev/null; then
        echo "wsl"
    else
        echo "linux"
    fi
}

ENV_TYPE=$(detect_env)

print_header() {
    echo "============================================"
    echo " SLM Eval - Termux/Android Benchmark"
    echo "============================================"
    echo "Environment:     $ENV_TYPE"

    if [ "$ENV_TYPE" = "termux" ]; then
        local model_name android_ver
        model_name=$(getprop ro.product.model 2>/dev/null || echo "unknown")
        android_ver=$(getprop ro.build.version.release 2>/dev/null || echo "unknown")
        echo "Device model:    $model_name"
        echo "Android version: $android_ver"
    else
        echo "Uname:           $(uname -a 2>/dev/null || echo unknown)"
    fi

    local cpu_cores
    cpu_cores=$(nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo 2>/dev/null || echo "unknown")
    echo "CPU cores:       $cpu_cores"

    if [ -f /proc/meminfo ]; then
        local total_ram_kb total_ram_gb
        total_ram_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        total_ram_gb=$(awk "BEGIN {printf \"%.2f\", $total_ram_kb/1024/1024}")
        echo "Total RAM:       ${total_ram_gb}GB"
    else
        echo "Total RAM:       unknown"
    fi

    local storage_avail
    storage_avail=$(df -h "$HOME" 2>/dev/null | awk 'NR==2 {print $4}')
    echo "Storage avail:   ${storage_avail:-unknown}"
    echo "============================================"
    echo
}

print_header

# ---------- dependency checks ----------
ensure_installed() {
    local cmd="$1"
    local pkg="$2"
    if command -v "$cmd" >/dev/null 2>&1; then
        return 0
    fi
    echo "Installing missing dependency: $pkg"
    if [ "$ENV_TYPE" = "termux" ]; then
        pkg install -y "$pkg"
    else
        sudo apt-get update -y && sudo apt-get install -y "$pkg"
    fi
}

if [ "$ENV_TYPE" = "termux" ]; then
    ensure_installed python python
    ensure_installed git git
    PYTHON_BIN=$(command -v python)
    "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || pkg install -y python-pip
else
    ensure_installed python3 python3
    ensure_installed git git
    PYTHON_BIN=$(command -v python3)
    "$PYTHON_BIN" -m pip --version >/dev/null 2>&1 || ensure_installed pip3 python3-pip
fi

# ---------- venv setup ----------
VENV_DIR="$HOME/slm-eval-env"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ---------- install slm_eval ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing slm_eval from $SCRIPT_DIR"
pip install --quiet --upgrade pip
pip install --quiet -e "$SCRIPT_DIR"

# ---------- run benchmark ----------
MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
CATEGORY="${2:-english_grammar}"
TIMEOUT="${3:-60}"

RESULTS_DIR="$HOME/slm_eval_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAFE_MODEL=$(echo "$MODEL" | tr '/: ' '___')
RESULT_FILE="$RESULTS_DIR/${SAFE_MODEL}_${TIMESTAMP}.json"

echo
echo "Running benchmark: model=$MODEL category=$CATEGORY timeout=${TIMEOUT}s per prompt"
echo

cd "$SCRIPT_DIR"
python -m slm_eval.termux_runner \
    --model "$MODEL" \
    --category "$CATEGORY" \
    --timeout-per-prompt "$TIMEOUT" \
    --output "$RESULT_FILE"

echo "Results saved to: $RESULT_FILE"
