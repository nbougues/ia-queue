#!/usr/bin/env python3
"""Gate d'un tick de cron Hermes : decide (sans LLM) s'il faut lancer une tache IA.

Appelle par le cron Hermes en mode `--no-agent` : **stdout vide = silencieux**.
Ce script ne dort JAMAIS (le scheduler cron attend sa fin) : il delègue le
« wait aleatoire 0-N s » a un worker detache.

Sequence complete (worker) : jitter -> GET /free -> POST /lease -> tache
Workflowy -> session Hermes -> pin -> release.

Usage : ia_slot_gate.py [--agent PROFIL] [--dry-run] [--sync]
  --sync   execute le worker en avant-plan (test manuel, bloque jusqu'a la fin)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ia_slot_common as C  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None, help="profil Hermes (auto-detecte sinon)")
    ap.add_argument("--dry-run", action="store_true", help="affiche la decision, ne lance rien")
    ap.add_argument("--sync", action="store_true", help="lance le worker en avant-plan")
    ap.add_argument("--no-jitter", action="store_true", help="passe le sommeil aleatoire (test)")
    args = ap.parse_args(argv)

    cfg = C.load_config()
    profile = C.agent_profile(args.agent)
    acfg = C.agent_conf(cfg, profile)

    def decide() -> tuple[bool, str]:
        if not acfg.get("enabled", True):
            return False, "agent %s desactive dans config.json" % profile
        ok, why = C.in_window(cfg)
        if not ok:
            return False, "hors fenetre (%s)" % why
        lock = C.AgentLock(profile)
        if not lock.acquire():
            return False, "un worker %s tourne deja" % profile
        lock.release()
        # la file d'abord : inutile de dormir 9 min pour rien (lecture du cache wf.py)
        pending, why_pending = C.has_pending(cfg, acfg["queue_path"], acfg)
        if not pending:
            return False, "file vide (%s)" % why_pending
        return True, why_pending

    ok, why = decide()
    C.log("tick agent=%s -> %s (%s)" % (profile, "go" if ok else "skip", why))
    if args.dry_run:
        print("gate %s: %s (%s)" % (profile, "lancerait un worker" if ok else "rien", why))
        return 0
    if not ok:
        return 0  # stdout vide => cron silencieux

    worker = [sys.executable, str(Path(__file__).resolve().parent / "ia_slot_worker.py"),
              "--agent", profile]
    if args.no_jitter:
        worker.append("--no-jitter")
    if args.sync:
        return subprocess.call(worker)

    C.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with C.log_path("worker.log").open("a", encoding="utf-8") as out:
        subprocess.Popen(worker, stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True, cwd=str(C.BASE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
