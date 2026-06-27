#!/usr/bin/env bash
# One-time setup for Billy's in-process emulator (stable-retro).
#
#   ./emulator/setup_retro.sh
#
# Creates a venv, installs deps, and imports the Super Mario Bros ROM into stable-retro's
# game-data folder (registers it as SuperMarioBros-Nes-v0). After this, just run:
#   .venv/bin/python run.py --attempts 8
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PYTHON:-python3.14}"
if [[ ! -d .venv ]]; then
  echo "[setup] creating venv with $PY"
  "$PY" -m venv .venv
fi

echo "[setup] installing dependencies"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

if [[ ! -f roms/smb.nes ]]; then
  echo "[setup] ERROR: place a legally-obtained 'Super Mario Bros (USA).nes' at roms/smb.nes" >&2
  exit 1
fi

echo "[setup] importing ROMs into stable-retro"
.venv/bin/python -m retro.import roms/

if [[ -f roms/zelda.nes ]]; then
  echo "[setup] Zelda ROM present (uses stable-retro's experimental LegendOfZeldaPRG0-Nes integration)"
  echo "[setup]   Run:  .venv/bin/python run.py --game zelda --attempts 5 --no-llm"
fi

echo "[setup] done. Run:  .venv/bin/python run.py --attempts 8"
