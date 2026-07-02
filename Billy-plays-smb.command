#!/bin/bash
# Billy-plays-smb.command — double-click to watch Billy Mitchell play Super Mario Bros.
# A "Billy Mitchell" window opens and he plays in real time (clears 1-1, 1-2, and crosses
# 1-3's tree-top section with his trained sub-policy). This Terminal window shows his play log.
# Close either window to stop.

REPO="/Users/jonathanbrasfield/repo/billy-mitchell"
cd "$REPO" || { echo "Can't find Billy at $REPO"; read -r; exit 1; }

if [ ! -x ".venv/bin/python" ]; then
  echo "No .venv found — run ./emulator/setup_retro.sh in $REPO first."
  read -r -p "Press Return to close…"
  exit 1
fi

echo "🕹️  Billy Mitchell is taking the controller… (close this window to stop)"
echo
# Watchable (windowed, real-time), pure learning loop, with the RL section sub-policies enabled.
.venv/bin/python run.py --attempts 6 --no-llm --rl-sections

echo
read -r -p "Billy is done. Press Return to close…"
