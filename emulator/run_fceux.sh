#!/usr/bin/env bash
# Launch FCEUX with the Billy bridge Lua script + the Super Mario Bros ROM.
#
# Usage:  ./emulator/run_fceux.sh [path/to/rom.nes]
# The Python brain (run.py) talks to this process over the lock-step files in $BILLY_RUNTIME.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROM="${1:-$REPO_ROOT/roms/smb.nes}"
BRIDGE="$REPO_ROOT/billy/systems/nes/bridge.lua"

# Shared IPC dir + emulation speed (normal so you can watch; set turbo/maximum for fast runs).
export BILLY_RUNTIME="${BILLY_RUNTIME:-$REPO_ROOT/.runtime}"
export BILLY_SPEED="${BILLY_SPEED:-normal}"
mkdir -p "$BILLY_RUNTIME"

if [[ ! -f "$ROM" ]]; then
  echo "ROM not found: $ROM" >&2
  echo "Place a legally-obtained 'Super Mario Bros (USA).nes' at roms/smb.nes" >&2
  exit 1
fi

echo "[run_fceux] runtime=$BILLY_RUNTIME speed=$BILLY_SPEED rom=$ROM"
exec fceux --loadlua "$BRIDGE" "$ROM"
