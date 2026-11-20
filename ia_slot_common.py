"""Helpers partages par ia_slot_gate.py et ia_slot_worker.py (stdlib uniquement).

Tout passe par ~/ia-queue/config.json ; aucun secret dans ces fichiers.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(os.environ.get("IA_QUEUE_HOME", str(Path.home() / "ia-queue")))
CONFIG_PATH = BASE / "config.json"
LOG_DIR = BASE / "logs"
STATE_DIR = BASE / "state"

# dossiers de la file (jamais des taches) + noms a ignorer en tete de file
FOLDER_NAMES = {"In Progress", "Done", "Archive", "AI Queue"}
# prefixe de titre des sessions lancees depuis la file
TITLE_PREFIX = "AI-QUEUE"

DEFAULTS = {
    "server": "http://inference-host:8765",
    "jitter_max_s": 540,
    "lease_ttl_s": 2700,
    "renew_every_s": 300,
    "max_task_minutes": 40,
    "queue_path": "Hermes-House/AI Queue",
    "runlog_path": "Hermes-House/Run Log",
    "wf": ["python3", str(Path.home() / "workflowy" / "wf.py")],
    "workdir": str(BASE),
    "window": {"start_hour": 0, "end_hour": 24, "days": "any"},
    "agents": {},
    "task_prompt_template": "",
}


def topic_from_task(text: str, limit: int = 58) -> str:
    """Sujet court, lisible, pour « AI-QUEUE <topic> » et le nom du livrable.

    La premiere proposition fait le sujet (« Faire X : detail... » -> « Faire X ») ;
    a defaut on tronque sur une frontiere de mot, jamais au milieu d'un mot.
    """
    clean = re.sub(r"<[^>]+>", " ", text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    clean = re.sub(r"^(EXEMPLE|TEST)\s*[—-]\s*", "", clean, flags=re.I)
    head = re.split(r"\s*[:—–|?]\s*", clean, maxsplit=1)[0].strip(" «»\"'.,;!")
    if len(head) < 12:                      # 1re proposition trop courte : prendre entier
        head = clean.strip(" «»\"'.,;!")
    if len(head) > limit:
        head = head[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return head or "tache"


# --------------------------------------------------------------------------- config

def load_config() -> dict:
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except FileNotFoundError:
        pass
    except (ValueError, OSError) as exc:
        raise SystemExit("config.json illisible (%s): %s" % (CONFIG_PATH, exc))
    if isinstance(cfg.get("wf"), str):
        cfg["wf"] = cfg["wf"].split()
    return cfg


def agent_profile(agent_arg: str | None) -> str:
    """Profil Hermes : --agent, sinon HERMES_HOME, sinon variable d'env, sinon 'default'."""
    if agent_arg:
        return agent_arg
    home = os.environ.get("HERMES_HOME", "")
    if home:
        parts = Path(home).parts
        if "profiles" in parts:
            return parts[parts.index("profiles") + 1]
        return "default"
    return os.environ.get("IA_QUEUE_AGENT", "default")


def agent_conf(cfg: dict, profile: str) -> dict:
    entry = dict(cfg["agents"].get(profile) or {})
    entry.setdefault("binary", "hermes" if profile == "default" else profile)
    entry.setdefault("label", profile)
    entry.setdefault("enabled", True)
    entry.setdefault("queue_path", cfg["queue_path"])
    entry.setdefault("runlog_path", cfg["runlog_path"])
    entry.setdefault("wf_home", None)   # WF_HOME de ce profil (identite Workflowy propre)
    entry.setdefault("check_free", True)  # False = ne pas attendre un GPU libre (modele distant)
    return entry


# --------------------------------------------------------------------------- fenetres

def in_window(cfg: dict, now: time.struct_time | None = None) -> tuple[bool, str]:
    win = cfg.get("window") or {}
    now = now or time.localtime()
    days = win.get("days", "any")
    if isinstance(days, str) and days != "any":
        allowed = {int(x) for x in days.replace("mon,1", "1").split(",") if x.strip().isdigit()}
        # time.tm_wday: 0 = lundi ; les jours ISO 1..7 (lun..dim)
        if ((now.tm_wday + 1) not in allowed) and (now.tm_wday not in allowed):
            return False, "day=%s" % (now.tm_wday + 1)
    start = int(win.get("start_hour", 0))
    end = int(win.get("end_hour", 24))
    hour = now.tm_hour
    ok = (start <= hour < end) if start <= end else (hour >= start or hour < end)
    return (ok, "hour=%s hors %s-%s" % (hour, start, end)) if not ok else (True, "")


# ---------------------------------------------------------------------------- HTTP

def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 8.0):
    """(status, body_dict) ; body_dict = {} si vide ou non JSON. N'leve pas sur 4xx/5xx."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(raw)
        except ValueError:
            return exc.code, {"raw": raw[:400]}
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return 0, {"error": "%s: %s" % (type(exc).__name__, exc)}


# --------------------------------------------------------------------------- logs

def log_path(name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / name


def log(line: str, name: str = "gate.log") -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path(name).open("a", encoding="utf-8") as handle:
        handle.write("%s %s\n" % (stamp, line))


def append_run(record: dict) -> None:
    record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "runs.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- locks

class AgentLock:
    """Un seul worker par agent ; le lock est libere quand le processus meurt."""

    def __init__(self, profile: str):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = STATE_DIR / ("%s.lock" % profile)
        self.handle = None

    def acquire(self) -> bool:
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            return False
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write("%d %s\n" % (os.getpid(), time.time()))
        self.handle.flush()
        return True

    def release(self) -> None:
        if self.handle:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_UN)
                self.handle.close()
            except OSError:
                pass
            self.handle = None


# ------------------------------------------------------------------------ workflowy

def wf_env(cfg: dict, agent_cfg: dict | None = None) -> dict:
    """Environnement des appels wf.py : WF_HOME par agent (identite Workflowy distincte)."""
    env = dict(os.environ)
    wf_home = (agent_cfg or {}).get("wf_home")
    if wf_home:
        env["WF_HOME"] = str(wf_home)
    return env


def wf(cfg: dict, args: list[str], timeout: int = 120, agent_cfg: dict | None = None):
    """Execute wf.py ; (rc, stdout, stderr)."""
    try:
        done = subprocess.run(cfg["wf"] + args, capture_output=True, text=True,
                              timeout=timeout, env=wf_env(cfg, agent_cfg))
        return done.returncode, done.stdout.strip(), done.stderr.strip()
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 124, "", str(exc)


def wf_json(cfg: dict, args: list[str], timeout: int = 120, agent_cfg: dict | None = None):
    rc, out, err = wf(cfg, args + ["--json"], timeout, agent_cfg)
    if rc != 0:
        return None, err or out
    try:
        return json.loads(out), ""
    except ValueError:
        return None, "sortie non JSON: %s" % out[:300]


def list_tasks(cfg: dict, queue_path: str, agent_cfg: dict | None = None):
    """(tasks, detail) — taches ouvertes de la file, dossiers exclus."""
    data, err = wf_json(cfg, ["todo", "list", "--path", queue_path], agent_cfg=agent_cfg)
    if data is None:
        return None, "wf todo list KO: %s" % err
    tasks = []
    for item in data.get("todos") or []:
        name = (item.get("name") or item.get("text") or "").strip()
        if not name or name in FOLDER_NAMES or name.startswith("#"):
            continue
        tasks.append({"id": item.get("id"), "name": name,
                      "note": (item.get("note") or "").strip()})
    return tasks, "ok"


def has_pending(cfg: dict, queue_path: str, agent_cfg: dict | None = None):
    """Gardien du tick : y a-t-1 au moins une tache en attente ? (lit le cache local)"""
    tasks, detail = list_tasks(cfg, queue_path, agent_cfg)
    if tasks is None:
        return False, detail
    return bool(tasks), "%d tache(s) en attente" % len(tasks)


# -------------------------------------------------------------------------- hermes

def run_hermes(binary: str, prompt: str, workdir: str, timeout_s: int,
               on_session=None) -> dict:
    """Lance `hermes chat -q <prompt> --oneshot -Q --format stream-json`.

    Distingue de `hermes -z` : celui-ci NE force PAS yolo, donc approvals.mode=smart
    s'applique (le gardien auxiliaire approuve le risqué-benin, `single_query_mode`
    bloque le reste au lieu de attendre un humain). Les evenements JSON sont lus en
    flux pour recuperer le session_id des qu'il apparait (`on_session`).
    """
    argv = [binary, "chat", "-q", prompt, "--oneshot", "-Q", "--format", "stream-json"]
    report = {"_rc": 1, "_session_id": "", "_text": "", "_tokens": {}, "_stderr_tail": "",
              "_elapsed_s": 0.0}
    started = time.time()
    err_file = None
    try:
        err_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(argv, cwd=workdir, stdout=subprocess.PIPE,
                                stderr=err_file, text=True, errors="replace", bufsize=1)
    except OSError as exc:
        report["_stderr_tail"] = "lancement %s impossible: %s" % (binary, exc)
        return report

    watchdog = threading.Timer(max(10, timeout_s), _kill, args=(proc,))
    watchdog.daemon = True
    watchdog.start()
    sid = ""
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    event = json.loads(line)
                except ValueError:
                    event = None
                if isinstance(event, dict):
                    sid = event.get("session_id") or sid
                    if event.get("type") == "result":
                        report["_text"] = event.get("text") or ""
                        report["_tokens"] = event.get("tokens") or {}
                        report["_rc"] = int(event.get("exit_code", 0))
                        report["_duration_ms"] = event.get("duration_ms")
            elif line.lower().startswith(("session:", "session_id:")):
                sid = line.split(":", 1)[1].strip() or sid
            if sid and sid != report["_session_id"]:
                report["_session_id"] = sid
                if on_session:
                    try:
                        on_session(sid)
                    except Exception:  # noqa: BLE001 — un callback capricieux ne tue pas la tache
                        pass
        proc.wait()
    finally:
        watchdog.cancel()
        try:
            proc.stdout.close()
        except (OSError, ValueError):
            pass
    err_file.seek(0)
    report["_stderr_tail"] = (err_file.read() or "")[-800:]
    err_file.close()
    report["_elapsed_s"] = round(time.time() - started, 1)
    report["completed"] = report["_rc"] == 0 and bool(report["_text"])
    report["total_tokens"] = (report.get("_tokens") or {}).get("total")
    return report


def _kill(proc) -> None:
    if proc.poll() is None:
        proc.kill()


def _session_cmd(binary: str, sub: str, session_id: str, extra: list[str] | None = None):
    if not session_id:
        return False, "pas de session_id"
    try:
        done = subprocess.run([binary, "sessions", sub, session_id] + (extra or []),
                              capture_output=True, text=True, timeout=60)
        return done.returncode == 0, (done.stdout or done.stderr).strip()[:300]
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)


def pin_session(binary: str, session_id: str) -> tuple[bool, str]:
    return _session_cmd(binary, "pin", session_id)


def rename_session(binary: str, session_id: str, title: str) -> tuple[bool, str]:
    return _session_cmd(binary, "rename", session_id, [title])
