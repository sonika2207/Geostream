#!/usr/bin/env bash
# Startup script for *nix
set -euo pipefail
python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
