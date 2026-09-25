#!/usr/bin/env bash
# run_server.sh - Automated launcher for the Live Subtitles GPU Server
# Automatically creates virtual environment and installs headless GPU requirements if needed.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "=================================================================="
    echo "Setting up Python virtual environment for GPU Server..."
    echo "=================================================================="
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "Upgrading pip and installing GPU server requirements..."
    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements-server.txt"
    echo "Setup complete!"
    echo "=================================================================="
else
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

python3 "$SCRIPT_DIR/server.py" "$@"
