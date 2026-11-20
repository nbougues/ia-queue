#!/usr/bin/env python3
"""Worker d'un creneau IA : dort un temps aleatoire (<= jitter_max_s), verifie la
liberte du serveur IA, prend un bail global, pioche la prochaine tache de la file
Workflowy de CE profil, la fait executer par une vraie session Hermes en mode
smart, epingle (pin) et nomme la session, puis rend le bail.

Lance en general par ia_slot_gate.py (detache). Tests manuels :

    python3 ia_slot_worker.py --agent default --no-jitter
    python3 ia_slot_worker.py --agent default --no-jitter --task "Recapitule X"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import time
from pathlib import Path
from string import Formatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ia_slot_common as C  # noqa: E402

# Le « prompt systeme » de la session lancee depuis la file. Ecrasable par la cle
# task_prompt_template de config.json (placeholders entre accolades).
DEFAULT_TEMPLATE = """# Contexte d'execution

Tu es l'agent Hermes « {agent_label} » (profil `{profile}`). Tu tournes **en \
arrière-plan, sans personne pour te répondre** : un créneau GPU t'a été accordé par \
le serveur ia-free (bail {lease_id}) et ta session sera épinglée dans la barre \
latérale pour relecture ultérieure.

# Nom de la tâche

Sujet retenu : **{topic}**
- Ta session est nommée `{title_prefix} {topic}` (déjà renommée par le worker).
- Nomme tout fichier que tu produis avec ce sujet, préfixé par la date : \
`{yymmdd}-{slug}.md` (slug = sujet en minuscules, sans accents ni espaces).

# Tâche (file « {queue_path} » dans Workflowy)

=== DÉBUT TÂCHE ===
{task}
=== FIN TÂCHE ===
{task_note}
# Règles impératives

1. **Tu tournes en mode unattended.** N'utilise jamais l'outil qui pose une question \
à l'utilisateur, n'attends aucune validation, ne laisse aucune étape en attente. Fais \
des **hypothèses raisonnables**, explicite-les, et va jusqu'au bout.
2. Rassemble les **questions que tu aurais posées** dans une section « QUESTIONS EN \
SUSPENS » du livrable (avec l'hypothèse retenue pour chacune) et reprends-les dans le \
BILAN. Écris « aucune » s'il n'y en a pas.
3. Les commandes dangereuses sont **refusées d'office** (mode smart : aucun humain \
disponible pour approuver). Si une commande est bloquée, n'insiste pas — contourne, ou \
note ce qui a été bloqué dans une section « BLOQUÉ ».
4. Tâche **non prioritaire** : aucune écriture destructive, aucun achat, aucun envoi \
de mail/message, aucune publication vers un tiers. Prépare le brouillon, laisse-le à \
relire.
5. Budget : **{max_task_minutes} minutes**. Approche de la limite → arrête-toi et \
rends ce qui est fait.
6. Écris le livrable dans `{workdir}/livrables/` si le résultat mérite d'être gardé.
7. Termine ton dernier message par :
   BILAN: <2-3 phrases : fait / reste / chemin du livrable / questions en suspens>
"""


class SafeFormatter(Formatter):
    def get_value(self, key, args, kwargs):  # noqa: D102
        if isinstance(key, str):
            return kwargs.get(key, "{%s}" % key)
        return Formatter.get_value(self, key, args, kwargs)


def claim_task(cfg: dict, task: dict, queue_path: str, acfg: dict) -> bool:
    if not task.get("id"):
        return False
    rc, out, err = C.wf(cfg, ["mv", task["id"], "%s/In Progress" % queue_path], agent_cfg=acfg)
    if rc != 0:
        C.log("claim KO (%s): %s" % (task["name"][:40], (err or out)[:200]))
        return False
    return True


def finish_task(cfg: dict, task: dict, queue_path: str, acfg: dict, ok: bool) -> None:
    if not task.get("id"):
        return
    if ok:
        C.wf(cfg, ["todo", "done", task["id"]], agent_cfg=acfg)
        C.wf(cfg, ["mv", task["id"], "%s/Done" % queue_path], agent_cfg=acfg)
    else:
        C.wf(cfg, ["mv", task["id"], queue_path], agent_cfg=acfg)  # retour dans la file


def write_runlog(cfg: dict, runlog_path: str, acfg: dict, line: str, note: str) -> None:
    rc, out, err = C.wf(cfg, ["capture", line, "--parent", runlog_path, "--note", note],
                        agent_cfg=acfg)
    if rc != 0:
        C.log("runlog KO: %s" % (err or out)[:200])


class LeaseHeartbeat(threading.Thread):
    """Renouvelle le bail pendant toute la duree de la tache."""

    def __init__(self, server: str, lease_id: str, ttl: int, every: int):
        super().__init__(daemon=True)
        self.server, self.lease_id, self.ttl, self.every = server, lease_id, ttl, every
        self._stop = threading.Event()
        self.lost = False

    def run(self) -> None:
        while not self._stop.wait(self.every):
            code, body = C.http_json("POST", "%s/lease/renew" % self.server,
                                      {"lease_id": self.lease_id, "ttl": self.ttl})
            if code != 200:
                self.lost = True
                C.log("bail perdu (%s): %s" % (code, body))
                return

    def stop(self) -> None:
        self._stop.set()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "tache"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", default=None)
    ap.add_argument("--no-jitter", action="store_true")
    ap.add_argument("--task", default=None, help="forcer une tache (ignore la file)")
    ap.add_argument("--force", action="store_true", help="ignore le verdict /free (test)")
    args = ap.parse_args(argv)

    cfg = C.load_config()
    profile = C.agent_profile(args.agent)
    acfg = C.agent_conf(cfg, profile)
    server = cfg["server"].rstrip("/")
    queue_path = acfg["queue_path"]
    run = {"agent": profile, "label": acfg["label"], "queue": queue_path, "outcome": "start"}

    lock = C.AgentLock(profile)
    if not lock.acquire():
        C.log("worker %s deja en cours, abandon" % profile, "worker.log")
        return 0
    try:
        # 1. sommeil aleatoire, borne a jitter_max_s (540 s = 9 min par defaut) -----
        delay = 0 if args.no_jitter else random.uniform(0, float(cfg["jitter_max_s"]))
        run["jitter_s"] = round(delay, 1)
        C.log("worker %s reveille, attend %.0fs" % (profile, delay), "worker.log")
        if delay:
            time.sleep(delay)

        # 2. liberte du serveur : 0 ou 1 tache en cours => on y va ------------------
        code, verdict = C.http_json("GET", "%s/free" % server)
        run["verdict"] = {"http": code, "reason": verdict.get("reason"),
                          "free": verdict.get("free"), "signals": verdict.get("signals")}
        C.log("verdict %s: %s" % (profile, json.dumps(run["verdict"], ensure_ascii=False)),
              "worker.log")
        if not args.force and acfg.get("check_free", True) and (code != 200 or not verdict.get("free")):
            run["outcome"] = "busy"
            return 0

        # 3. bail global ------------------------------------------------------------
        ttl = int(cfg["lease_ttl_s"])
        code, lease = C.http_json("POST", "%s/lease" % server,
                                   {"agent": acfg["label"], "task": "claiming", "ttl": ttl})
        if code != 200:
            run["outcome"] = "lease_denied"
            run["lease_denied"] = lease
            C.log("bail refuse pour %s: %s" % (profile, lease), "worker.log")
            return 0
        lease_id = lease["id"]
        run["lease_id"] = lease_id

        hb = LeaseHeartbeat(server, lease_id, ttl, int(cfg["renew_every_s"]))
        hb.start()
        try:
            # 4. tache ---------------------------------------------------------------
            if args.task:
                task = {"id": None, "name": args.task, "note": ""}
            else:
                tasks, why = C.list_tasks(cfg, queue_path, acfg)
                if tasks is None:
                    run["outcome"] = "queue_error"
                    run["queue_detail"] = why
                    C.log("lecture file KO pour %s: %s" % (profile, why), "worker.log")
                    return 0
                if not tasks:
                    run["outcome"] = "queue_empty"
                    C.log("file vide pour %s" % profile, "worker.log")
                    return 0
                task = tasks[0]
                if not task["note"] and task["id"]:
                    note, _ = C.wf_json(cfg, ["note", "get", task["id"]], agent_cfg=acfg)
                    if isinstance(note, dict):
                        task["note"] = (note.get("note") or "").strip()
            topic = C.topic_from_task(task["name"])
            title = "%s %s" % (C.TITLE_PREFIX, topic)
            run.update({"task": task["name"][:200], "task_id": task.get("id"),
                        "topic": topic, "title": title})
            if task.get("id"):
                claim_task(cfg, task, queue_path, acfg)
            C.http_json("POST", "%s/lease" % server,
                        {"agent": acfg["label"], "task": task["name"][:300], "ttl": ttl})

            # 5. execution par une vraie session Hermes, epinglee des l'identification
            template = cfg.get("task_prompt_template") or DEFAULT_TEMPLATE
            prompt = SafeFormatter().format(
                template,
                task=task["name"] + ("\n\n" + task["note"] if task["note"] else ""),
                task_note=("NOTE DE LA TÂCHE (Workflowy) :\n%s\n\n" % task["note"]) if task["note"] else "",
                agent_label=acfg["label"], profile=profile, queue_path=queue_path,
                lease_id=lease_id, max_task_minutes=cfg["max_task_minutes"],
                workdir=cfg["workdir"], topic=topic, title_prefix=C.TITLE_PREFIX,
                yymmdd=time.strftime("%y%m%d"), slug=slugify(topic))

            def on_session(sid: str) -> None:
                renamed, _ = C.rename_session(acfg["binary"], sid, title)
                pinned, _ = C.pin_session(acfg["binary"], sid)
                run["pinned_early"] = pinned
                run["renamed_early"] = renamed
                C.log("session %s -> « %s » pin=%s rename=%s"
                      % (sid[:12], title, pinned, renamed), "worker.log")

            report = C.run_hermes(acfg["binary"], prompt, cfg["workdir"],
                                  int(cfg["max_task_minutes"]) * 60 + 120, on_session)
            session_id = report.get("_session_id") or ""
            run.update({"session_id": session_id, "rc": report.get("_rc"),
                        "elapsed_s": report.get("_elapsed_s"),
                        "completed": report.get("completed"),
                        "total_tokens": report.get("total_tokens"),
                        "answer_tail": (report.get("_text") or "")[-600:]})
            C.log("session %s terminee rc=%s duree=%ss"
                  % (session_id[:12], report.get("_rc"), report.get("_elapsed_s")),
                  "worker.log")

            # 6. filet de securite si le flux n'a pas donne l'identifiant
            if session_id and not run.get("pinned_early"):
                run["pinned"], run["pin_detail"] = C.pin_session(acfg["binary"], session_id)
                run["renamed"], _ = C.rename_session(acfg["binary"], session_id, title)

            # 7. ranger la tache + journal -------------------------------------------
            ok = report.get("_rc") == 0 and bool(report.get("completed"))
            run["outcome"] = "done" if ok else "failed"
            if not args.task:
                finish_task(cfg, task, queue_path, acfg, ok)
            summary = "[{}] {} / {} : {} — {}".format(
                time.strftime("%Y-%m-%d"), acfg["label"], C.TITLE_PREFIX, topic,
                "OK" if ok else "ECHEC (rc=%s)" % report.get("_rc"))
            note = "session=%s\ntitre=%s\nepinglee=%s\nduree=%ss\ntokens=%s\n\n%s" % (
                session_id, title, run.get("pinned_early") or run.get("pinned"),
                report.get("_elapsed_s"), report.get("total_tokens"),
                (report.get("_text") or report.get("_stderr_tail") or "")[-1200:])
            write_runlog(cfg, acfg["runlog_path"], acfg, summary, note)
            return 0
        finally:
            hb.stop()
            code, body = C.http_json("DELETE", "%s/lease/%s" % (server, lease_id))
            run["lease_released"] = code == 200
    finally:
        run["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        C.append_run(run)
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
