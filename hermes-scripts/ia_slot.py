#!/usr/bin/env python3
"""Wrapper installe dans <HERMES_HOME>/scripts/ia_slot.py par un agent Hermes.

Le scheduler cron n'accepte que des scripts situes dans le dossier scripts/ DU
PROFIL (pas de symlink pointant ailleurs), d'ou ce fichier minimal. Le vrai code
vit dans ~/ia-queue/ (partage par les 3 agents).

Le profil est devine depuis l'emplacement du wrapper :
  ~/.hermes/scripts/ia_slot.py                      -> default
  ~/.hermes/profiles/<profil>/scripts/ia_slot.py    -> <profil>
"""

import subprocess
import sys
from pathlib import Path

BASE = Path.home() / "ia-queue"

profile = "default"
parts = Path(__file__).resolve().parts
if "profiles" in parts and parts.index("profiles") + 1 < len(parts):
    profile = parts[parts.index("profiles") + 1]

sys.exit(subprocess.call(
    [sys.executable, str(BASE / "ia_slot_gate.py"), "--agent", profile],
    cwd=str(BASE)))
