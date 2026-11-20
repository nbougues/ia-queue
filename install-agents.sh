#!/usr/bin/env bash
# Installe le creneau IA dans un ou plusieurs profils Hermes de CETTE machine :
#   1. copie le wrapper ia_slot.py dans <HERMES_HOME>/scripts/ (obligatoire :
#      le scheduler cron refuse tout script hors de ce dossier, symlink compris)
#   2. cree le job cron "creneau-ia" (mode --no-agent : 0 token quand le GPU est occupe)
#
#   ./install-agents.sh                 # default ops finance
#   ./install-agents.sh default         # un seul profil
#   SCHEDULE="every 2h" ./install-agents.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULE="${SCHEDULE:-every 10m}"
JOB_NAME="${JOB_NAME:-creneau-ia}"
PROFILES="${*:-default finance}"

for p in $PROFILES; do
  if [ "$p" = "default" ]; then
    BIN="hermes"; SDIR="$HOME/.hermes/scripts"
  else
    BIN="$p"; SDIR="$HOME/.hermes/profiles/$p/scripts"
  fi
  if ! command -v "$BIN" >/dev/null 2>&1; then
    echo "!! profil $p : binaire '$BIN' introuvable, ignore"; continue
  fi
  echo "== profil $p ($BIN) =="
  mkdir -p "$SDIR"
  install -m 0755 "$HERE/hermes-scripts/ia_slot.py" "$SDIR/ia_slot.py"
  echo "   wrapper -> $SDIR/ia_slot.py"

  if "$BIN" cron list 2>/dev/null | grep -q "$JOB_NAME"; then
    echo "   job '$JOB_NAME' deja present (utiliser: $BIN cron list / cron edit)"
    continue
  fi
  "$BIN" cron create --name "$JOB_NAME" --no-agent --script ia_slot.py \
      --deliver local --catch-up off "$SCHEDULE"
  echo "   job cree : $SCHEDULE"
done

echo
echo "Verification : hermes cron list   |   python3 $HERE/ia_slot_gate.py --agent default --dry-run"
