#!/usr/bin/env python3
"""ia-free-server — « le serveur IA est-il libre ? » + bail (lease) GPU global.

Deploye sur inference-host (inference-host, inference-host). **Sans authentification** :
concu pour un reseau prive uniquement, ne pas exposer au WAN.

Sources de verite (locales a la machine) :
  * http://127.0.0.1:8731/metrics  halogen/llama.cpp : requests_processing,
    requests_deferred, kv_cache_usage_ratio, ...
  * http://127.0.0.1:8731/slots    nombre total de slots de session
  * /sys/class/drm/card*/device/gpu_busy_percent  occupation GPU (0-100), informatif

Regle « libre » (volontairement permissive) :
  0 ou 1 requete d'inférence en cours  -> on y va
  2 requetes ou plus                   -> on n'y va pas
  (IA_FREE_MAX_INFLIGHT pour changer la limite, IA_FREE_COOLDOWN_S=0 par defaut,
   IA_FREE_MAX_GPU=-1 : le %% GPU n'est pas un critere de refus)
  Plus un bail global : un seul agent Hermes ne travaille a la fois.

Oisivete (`idle_for_s`) : secondes ecoulees depuis la derniere trace d'activite du
moteur (TOUTE requete, y compris une requete solitaire). Distinct de `calm_for_s`,
qui mesure depuis la derniere CONTENTION (plus de max_inflight requetes) et ne bouge
donc pas sur une requete seule. Un creneau « carte calme depuis 30 s » se lit sur
`idle_for_s` / `idle_ok`, pas sur `calm_for_s`.

Pause globale : elle met le service en veille sans toucher au moteur. Pendant une
pause, /free repond `free=false, reason="paused", idle_ok=false`, mais les deux
horloges d'oisivete (`calm_for_s`, `idle_for_s`) continuent de tourner normalement :
la pause est un verdict, pas une activite. Elle est persistee comme le bail, donc
elle survit a un restart du service.

Endpoints :
  GET    /free            verdict : {"free": bool, "reason": str, "idle_for_s": float,
                                     "idle_ok": bool, "signals": {...}, "pause": {...}}
  GET    /status          /free + configuration active
  GET    /pause           etat de la pause : {"paused": bool, "until": float|null,
                                              "reason": str, "remaining_s": float|null}
  POST   /pause           {"until": float|null, "reason": str} -> 200 {pause} | 400
  POST   /resume          leve la pause -> 200 {pause}
  GET    /lease           bail courant (ou null)
  POST   /lease           {"agent":str,"task":str,"ttl":int} -> 200 {lease_id,...} | 409
  POST   /lease/renew     {"lease_id":str,"ttl":int}         -> 200 | 410
  DELETE /lease/<id>      libere le bail (410 s'il ne correspond pas)
  GET    /history         N derniers evenements (JSON)
  GET    /healthz         "ok"

Code : stdlib uniquement (Python >= 3.9), aucune dependance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ----------------------------------------------------------------------------- config

def _env(name, default, cast=None):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return cast(raw) if cast else raw


CONFIG = {
    "bind": _env("IA_FREE_BIND", "0.0.0.0"),
    "port": int(_env("IA_FREE_PORT", "8765")),
    "metrics_url": _env("IA_FREE_METRICS_URL", "http://127.0.0.1:8731/metrics"),
    "slots_url": _env("IA_FREE_SLOTS_URL", "http://127.0.0.1:8731/slots"),
    "gpu_glob": _env("IA_FREE_GPU_GLOB", "/sys/class/drm/card*/device/gpu_busy_percent"),
    # seuil d'occupation GPU bloquant ; -1 = desactive (le %% GPU reste un signal
    # informatif : il plafonne a 100 % des que la carte sert, ce n'est pas un refus)
    "gpu_max": float(_env("IA_FREE_GPU_MAX", "-1")),
    # nombre de requetes d'inférence acceptees pour dire « on y va ». Ce n'est PLUS
    # la contrainte principale depuis que le moteur sert plusieurs requetes de front
    # (4 slots, jusqu'a 8 flux en rounds conjoints) : c'est un simple garde-fou de
    # file d'attente. La vraie capacite est mesuree par kv_reserve_positions ci-dessous.
    "max_inflight": int(_env("IA_FREE_MAX_INFLIGHT", "3")),
    # duree (s) d'activite nulle exigee avant de declarer libre ; 0 = aucun cooldown
    "cooldown_s": float(_env("IA_FREE_COOLDOWN_S", "0")),
    # periode (s) d'echantillonnage du moteur par le sondeur d'oisivete ; 0 = desactive
    # (idle_for_s n'est plus recalcule qu'aux appels clients, donc moins precis)
    "idle_sample_s": float(_env("IA_FREE_IDLE_SAMPLE_S", "2")),
    # calme (s) sans aucune requete exige par le verdict « idle_ok » (les clients
    # libres de l'ignorer : idle_for_s est brut, idle_ok est la regle convenue)
    "idle_min_s": float(_env("IA_FREE_IDLE_MIN_S", "30")),
    "engine_timeout_s": float(_env("IA_FREE_ENGINE_TIMEOUT", "3")),
    "lease_ttl_s": int(_env("IA_FREE_LEASE_TTL", "2700")),
    "lease_max_ttl_s": int(_env("IA_FREE_LEASE_MAX_TTL", "14400")),
    "state_file": _env("IA_FREE_STATE_FILE", "/var/lib/ia-free/state.json"),
    "history_size": int(_env("IA_FREE_HISTORY", "200")),
    # --- chaleur de cache (observation : ces regles ne changent PAS le verdict) ---
    # une session est dite « chaude » si son dernier ratio de cache est bon ET qu'elle
    # a ete vue recemment ET que personne n'a evince depuis (compteur moteur partage).
    "warm_grace_s": float(_env("IA_FREE_WARM_GRACE_S", "90")),
    "warm_ratio_min": float(_env("IA_FREE_WARM_RATIO_MIN", "0.5")),
    # une session vue il y a moins de protect_age_s est « en cours » : son empreinte KV
    # n'est pas comptee comme recuperable. Au-dela, son cache est froid et evictable.
    "protect_age_s": float(_env("IA_FREE_PROTECT_AGE_S", "60")),
    # au-dela, l'entree est oubliee (une instance morte ne doit pas proteger le vide)
    "warm_ttl_s": float(_env("IA_FREE_WARM_TTL_S", "900")),
    "warm_max_sessions": int(_env("IA_FREE_WARM_MAX_SESSIONS", "200")),
    # --- budget en positions KV : LA vraie contrainte, pas le nombre de requetes ---
    # Le drame mesure d'avant-reglage : 5 sessions de 148k pour un pool de 786k, 0 % de
    # retention, et chaque tour repayait un prefill complet (~88 s pour 148k tokens).
    # Une eviction ne coute pas « un peu moins vite », elle coute une re-lecture GPU
    # integrale du contexte. La bonne question n'est donc pas « combien de requetes
    # tournent » mais « reste-t-il de quoi loger le contexte de celle qui arrive sans
    # entamer celles qui sont chaudes ».
    # L'arbitre ne connait pas la taille de la requete a venir (elle n'est pas encore
    # recue) : il reserve la place d'UN contexte de taille typique, et refuse si ce
    # n'est pas disponible. Fail-open si le moteur ne publie pas ces compteurs.
    "kv_gate": _env("IA_FREE_KV_GATE", "1", bool),
    "kv_reserve_positions": int(_env("IA_FREE_KV_RESERVE_POSITIONS", "157000")),
}

# --------------------------------------------------------------------------- etat

_LOCK = threading.Lock()
STATE = {
    "lease": None,               # {id, agent, task, acquired_at, ttl, expires_at, renewals}
    # pause globale : {until: float|None (None = indefinite), reason: str, set_at: float}
    "pause": None,
    "last_activity_ts": time.time(),
    "last_activity_reason": "boot",
    # oisivete du moteur : derniere trace d'activite vue (voir track_idle)
    "last_busy_ts": time.time(),
    "last_busy_reason": "boot",
    "requests_total_seen": None,
    # generation du moteur : +1 a chaque redemarrage detecte (compteurs qui repartent
    # de zero). La chaleur de cache n'a aucune valeur d'une generation a l'autre.
    "engine_epoch": 0,
    # chaleur de cache, par session : {ts, prompt_tokens, cached_tokens, ratio,
    # tag, evicted_at_report}. Alimente par POST /warmth depuis CHAQUE instance
    # Hermes : c'est l'arbitre qui voit tout le monde, pas chaque client qui ne voit
    # que ses propres appels.
    "warmth": {},
    "started_at": time.time(),
}
HISTORY = deque(maxlen=CONFIG["history_size"])
_last_persist = 0.0


def _persist(force: bool = False) -> None:
    """Ecrit l'etat (bail + horodatage d'activite) pour survivre a un redemarrage."""
    global _last_persist
    now = time.time()
    if not force and (now - _last_persist) < 2.0:
        return
    _last_persist = now
    path = Path(CONFIG["state_file"])
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        payload = {
            "lease": STATE["lease"],
            "pause": STATE.get("pause"),
            "last_activity_ts": STATE["last_activity_ts"],
            "last_activity_reason": STATE["last_activity_reason"],
            "last_busy_ts": STATE["last_busy_ts"],
            "last_busy_reason": STATE["last_busy_reason"],
            "warmth": STATE.get("warmth"),
            "engine_epoch": STATE.get("engine_epoch", 0),
            # la reference de compteurs survit aussi : sans elle, un redemarrage du
            # moteur pendant que l'arbitre est arrete serait invisible au premier
            # echantillon (rien a comparer).
            "requests_total_seen": STATE.get("requests_total_seen"),
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:  # journalisable, jamais fatal
        log("state_write_error", error=str(exc))


def _restore() -> None:
    path = Path(CONFIG["state_file"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(payload.get("lease"), dict):
        STATE["lease"] = payload["lease"]
    if isinstance(payload.get("pause"), dict):
        STATE["pause"] = payload["pause"]
    if isinstance(payload.get("warmth"), dict):
        # une table perdue n'est pas grave : tout le monde parait froid un appel, puis
        # se rallume des le premier rapport. Rien de cassé, juste un appel plus lent.
        STATE["warmth"] = payload["warmth"]
    if isinstance(payload.get("engine_epoch"), int):
        STATE["engine_epoch"] = payload["engine_epoch"]
    if payload.get("requests_total_seen") is not None:
        STATE["requests_total_seen"] = payload["requests_total_seen"]
    for key in ("last_activity_ts", "last_activity_reason",
                "last_busy_ts", "last_busy_reason"):
        if key in payload:
            STATE[key] = payload[key]


def log(event: str, **fields) -> None:
    entry = {"ts": round(time.time(), 3), "event": event}
    entry.update(fields)
    HISTORY.append(entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


# --------------------------------------------------------------------- signaux moteur

def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers={"Accept": "text/plain, application/json"})
    with urllib.request.urlopen(req, timeout=CONFIG["engine_timeout_s"]) as resp:
        return resp.read().decode("utf-8", "replace")


def _parse_prometheus(text: str) -> dict:
    series = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            series[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return series


def _metric(series: dict, name: str) -> float | None:
    """Valeur d'une serie Prometheus, `None` si absente (jamais KeyError).

    Tolere les series portées par des labels (`name{model="x"} 42`), que
    `_parse_prometheus` indexe telles quelles.
    """
    if name in series:
        return series[name]
    prefix = name + "{"
    for key, value in series.items():
        if key.startswith(prefix):
            return value
    return None


def read_gpu_busy() -> float | None:
    """Occupation GPU %% via sysfs amdgpu (max sur toutes les cartes)."""
    import glob

    worst = None
    for path in sorted(glob.glob(CONFIG["gpu_glob"])):
        try:
            value = float(Path(path).read_text().strip())
        except (OSError, ValueError):
            continue
        worst = value if worst is None else max(worst, value)
    return worst


def read_engine() -> dict:
    """Signaux du moteur d'inférence. `reachable=False` si le moteur ne répond pas."""
    out = {"reachable": False, "requests_processing": None, "requests_deferred": None,
           "inflight": None, "slots_total": None, "kv_cache_ratio": None,
           "gpu_busy_percent": None, "busy_for_seconds": None, "requests_total": None,
           # signaux de performance : pure information, jamais un critere de refus.
           # Une serie absente reste a None, et ne rend jamais le moteur « injoignable ».
           "decode_tps": None, "prefill_tps": None, "cache_hit_ratio": None,
           "queue_wait_seconds_total": None, "queue_wait_longest_seconds": None,
           # sante du cache de prefixe et du pool KV (voir read_engine)
           "cache_evicted_total": None, "cache_hits_total": None,
           "cache_misses_total": None, "cache_entries": None,
           "kv_pool_positions": None, "kv_cache_tokens": None}
    try:
        series = _parse_prometheus(_http_get(CONFIG["metrics_url"]))
    except Exception as exc:  # noqa: BLE001 - moteur hors service = info, pas un crash
        out["engine_error"] = "%s: %s" % (type(exc).__name__, exc)
        out["gpu_busy_percent"] = read_gpu_busy()
        return out
    out["reachable"] = True
    for src, dst in (("llamacpp:requests_processing", "requests_processing"),
                     ("llamacpp:requests_deferred", "requests_deferred"),
                     ("llamacpp:kv_cache_usage_ratio", "kv_cache_ratio"),
                     # halogen : streak d'occupation courant + compteur monotone de requetes
                     ("halogen:busy_for_seconds", "busy_for_seconds"),
                     ("halogen:requests_total", "requests_total")):
        if src in series:
            out[dst] = series[src]
    # debits instantanes rapportes par llama.cpp (gauges) + attente cumulee de la file
    # halogen : tout est lu sur LE MEME scrape /metrics, aucune requete supplementaire.
    for src, dst in (("llamacpp:predicted_tokens_seconds", "decode_tps"),
                     ("llamacpp:prompt_tokens_seconds", "prefill_tps"),
                     ("halogen:queue_wait_seconds_total", "queue_wait_seconds_total"),
                     ("halogen:queue_wait_longest_seconds", "queue_wait_longest_seconds")):
        out[dst] = _metric(series, src)
    # Taux de hits du cache de prefixe. `llamacpp:prompt_tokens_total` compte les
    # tokens de prompt REELLEMENT TRAITES, et `halogen:prompt_tokens_cached_total`
    # ceux que le cache a couverts (donc non traites) : les deux sont des parts d'un
    # meme total, le denominateur est leur somme, pas l'un des deux. (cached/prompt
    # donne ~10 sur ce moteur, evidemment faux.) Verifie : cached/(cached+prompt)
    # = 0,91344 ici, soit exactement le `token_hit_rate` que le moteur publie sur
    # /cache. Cumule depuis le demarrage, donc pas un ratio instantane.
    cached = _metric(series, "halogen:prompt_tokens_cached_total")
    prompt_total = _metric(series, "llamacpp:prompt_tokens_total")
    if cached is not None and prompt_total is not None and (cached + prompt_total) > 0:
        out["cache_hit_ratio"] = max(0.0, min(1.0, cached / (cached + prompt_total)))
    # Sante du cache de prefixe : le compteur d'evictions est LE signal partage entre
    # instances Hermes. Elles ne se connaissent pas entre elles, mais elles lisent
    # toutes le meme compteur : s'il bouge pendant qu'une session est inactive, son
    # entree est probablement partie. Aucune coordonnation Hermes<->Hermes necessaire.
    for src, dst in (("halogen:cache_evicted_total", "cache_evicted_total"),
                     ("halogen:cache_hits_total", "cache_hits_total"),
                     ("halogen:cache_misses_total", "cache_misses_total"),
                     ("halogen:cache_entries", "cache_entries"),
                     ("halogen:kv_pool_positions", "kv_pool_positions")):
        out[dst] = _metric(series, src)
    out["kv_cache_tokens"] = _metric(series, "llamacpp:kv_cache_tokens")
    try:
        out["slots_total"] = len(json.loads(_http_get(CONFIG["slots_url"])))
    except Exception:  # noqa: BLE001
        out["slots_total"] = None
    out["gpu_busy_percent"] = read_gpu_busy()
    return out


# --------------------------------------------------------------------- oisivete moteur

def track_idle(signals: dict, now: float) -> None:
    """Recale l'horloge d'oisivete : `idle_for_s` = temps ecoule depuis la derniere
    trace d'activite du moteur. Une trace = requete en vol (`requests_processing` +
    `requests_deferred` > 0), streak d'occupation rapporte par halogen
    (`busy_for_seconds` > 0), ou increment du compteur monotone `requests_total`
    (une requete s'est terminee entre deux sondeos). Moteur injoignable = occupe :
    sans signal on ne peut affirmer que la carte est libre.

    A ne pas confondre avec `last_activity_ts` (verdict « libre ») : celui-ci ne se
    recale que sur une CONTENTION (plus de `max_inflight` requetes), donc une requete
    solitaire ne le bouge pas ; `last_busy_ts` compte TOUTE activite, seul critere
    d'un creneau « la carte n'a rien servi depuis N secondes ».
    """
    reason = None
    if not signals.get("reachable"):
        reason = "engine_unreachable"
    else:
        inflight = (signals.get("requests_processing") or 0) + (signals.get("requests_deferred") or 0)
        if inflight > 0:
            reason = "requests_inflight"
        elif (signals.get("busy_for_seconds") or 0) > 0:
            reason = "busy_streak"
    with _LOCK:
        total = signals.get("requests_total")
        if total is not None:
            seen = STATE.get("requests_total_seen")
            if seen is not None and total < seen:
                note_engine_restart(seen, total, signals)
            if seen is not None and total > seen:
                reason = reason or "request_completed"
            STATE["requests_total_seen"] = total
        # Deuxieme garde, sans reference memorisee : une session dont le rapport
        # porte un compteur d'evictions PLUS ELEVE que celui du moteur prouve que les
        # compteurs ont reparte de zero. Le moteur a donc redemarre entre-temps, meme
        # si l'arbitre etait lui-meme arrete a ce moment-la (aucun `seen` a comparer).
        ev_now = signals.get("cache_evicted_total")
        if ev_now is not None and STATE.get("warmth"):
            reported = [float(row["evicted_at_report"])
                        for row in STATE["warmth"].values()
                        if isinstance(row, dict) and row.get("evicted_at_report") is not None]
            if reported and ev_now < min(reported):
                note_engine_restart(None, total, signals)
        if reason:
            STATE["last_busy_ts"] = now
            STATE["last_busy_reason"] = reason


def note_engine_restart(seen, total, signals: dict) -> None:
    """Le moteur a redemarre : ses compteurs repartent de zero et son cache KV est vide.

    Sans ca, la table de chaleur devient un piège : `evicted_since` est calcule en
    delta du compteur d'evictions, un compteur qui a décrémenté donne `max(0, …) = 0`,
    donc « personne n'a évidé », donc une session déclarée chaude juste avant le
    redémarrage reste **chaude** alors que le moteur a tout perdu. Le seul remède est
    l'invalidation : on ne peut pas savoir ce qui survit, et rien ne survit.

    Appelé avec `_LOCK` tenu.
    """
    epoch = int(STATE.get("engine_epoch") or 0) + 1
    dropped = len(STATE.get("warmth") or {})
    STATE["engine_epoch"] = epoch
    STATE["warmth"] = {}
    if total is not None:
        STATE["requests_total_seen"] = total
    log("engine_restart", previous_requests_total=seen, requests_total=total,
        epoch=epoch, warmth_dropped=dropped,
        evicted_total=signals.get("cache_evicted_total"))
    _persist(force=True)


def idle_sampler(stop: threading.Event) -> None:
    """Sonde le moteur en continu pour qu'`idle_for_s` progresse meme quand aucun
    client n'appelle /free (sinon l'horloge ne avancerait qu'aux sondeos clients)."""
    while not stop.wait(CONFIG["idle_sample_s"]):
        try:
            track_idle(read_engine(), time.time())
        except Exception as exc:  # noqa: BLE001 - un sondeur ne doit jamais tuer le service
            log("idle_sample_error", error="%s: %s" % (type(exc).__name__, exc))


# ------------------------------------------------------------------------- verdict

def lease_active(now: float | None = None) -> dict | None:
    lease = STATE.get("lease")
    if not lease:
        return None
    if float(lease.get("expires_at", 0)) <= (now or time.time()):
        return None
    return lease


def kv_headroom(signals: dict) -> dict:
    """Place libre dans le pool KV, et verdict « reste-t-il la place d'un contexte ? ».

    Le reserve est une taille de contexte typique, pas une mesure : la requete qui
    demande n'est pas encore parvenue a l'arbitre, qui ne peut donc pas connaitre sa
    taille. Sous la reserve, admettre une session de plus signifierait evincer un
    contexte resident chaud, c'est-a-dire payer un re-prefill complet (~88 s mesures
    pour 148k tokens) au lieu de servir en ~0,4 s.

    Fail-open : sans compteurs moteur, on admet. Un arbitre qui ne peut pas mesurer
    n'a pas le droit de bloquer du travail.
    """
    reserve = int(CONFIG["kv_reserve_positions"])
    pool = signals.get("kv_pool_positions")
    used = signals.get("kv_cache_tokens")
    if not pool or used is None:
        return {"kv_free_positions": None, "kv_reserve_positions": reserve, "kv_ok": True}
    free = int(pool) - int(used)
    return {"kv_free_positions": free, "kv_reserve_positions": reserve, "kv_ok": free >= reserve}


def evaluate(session: str = "", tokens=None, signals: dict | None = None) -> dict:
    now = time.time()
    signals = signals if signals is not None else read_engine()
    track_idle(signals, now)
    idle_for = now - float(STATE["last_busy_ts"])
    busy_reasons = []

    if not signals["reachable"]:
        busy_reasons.append("engine_unreachable")
    else:
        # regle demandee : 0 ou 1 requete en cours => on y va ; 2 ou plus => on n'y va pas
        inflight = int(signals["requests_processing"] or 0) + int(signals["requests_deferred"] or 0)
        signals["inflight"] = inflight
        if inflight > CONFIG["max_inflight"]:
            busy_reasons.append("inflight_above_limit")
    # budget KV : la capacite reelle, mesuree en positions libres du pool. Independant
    # de l'etat « joignable » ci-dessus : un moteur peut etre parfaitement calme et ne
    # plus avoir de place pour un contexte de plus (cas exact des 5 sessions de 148k).
    kv = kv_headroom(signals)
    signals.update(kv)
    if CONFIG["kv_gate"] and kv["kv_free_positions"] is not None and not kv["kv_ok"]:
        busy_reasons.append("kv_tight")
    gpu = signals["gpu_busy_percent"]
    if CONFIG["gpu_max"] >= 0 and gpu is not None and gpu >= CONFIG["gpu_max"]:
        busy_reasons.append("gpu_busy")

    if busy_reasons:
        STATE["last_activity_ts"] = now
        STATE["last_activity_reason"] = ",".join(busy_reasons)
        _persist()
    calm_for = now - float(STATE["last_activity_ts"])

    lease = lease_active(now)
    pause = pause_active(now)
    if pause:
        # la pause est un verdict, pas une activite : elle ne touche ni `calm_for_s`
        # ni `idle_for_s`, qui continuent de suivre le moteur normalement.
        reason = "paused"
    elif lease:
        reason = "lease_held:%s" % lease.get("agent")
    elif busy_reasons:
        reason = busy_reasons[0]
    elif CONFIG["cooldown_s"] > 0 and calm_for < CONFIG["cooldown_s"]:
        reason = "cooldown"
    elif not signals["reachable"]:
        reason = "engine_unreachable"
    else:
        reason = "free"

    return {
        "free": reason == "free",
        "reason": reason,
        "calm_for_s": round(calm_for, 1),
        "idle_for_s": round(idle_for, 1),
        "idle_ok": pause is None and idle_for >= CONFIG["idle_min_s"],
        "idle_min_s": CONFIG["idle_min_s"],
        "idle_reason": STATE["last_busy_reason"],
        "cooldown_s": CONFIG["cooldown_s"],
        "max_inflight": CONFIG["max_inflight"],
        "signals": {k: signals[k] for k in
                    ("reachable", "requests_processing", "requests_deferred", "inflight",
                     "slots_total", "kv_cache_ratio", "gpu_busy_percent",
                     "busy_for_seconds", "requests_total",
                     # signaux de performance (informatifs, jamais un critere de refus)
                     "decode_tps", "prefill_tps", "cache_hit_ratio",
                     "queue_wait_seconds_total", "queue_wait_longest_seconds",
                     # sante du cache : `cache_evicted_total` est le signal partage
                     # entre instances Hermes, celles-ci ne se connaissent pas entre
                     # elles mais lisent toutes le meme compteur
                     "cache_evicted_total", "cache_hits_total", "cache_misses_total",
                     "cache_entries", "kv_pool_positions", "kv_cache_tokens",
                     # budget KV : `kv_ok=False` => plus la place d'un contexte de plus
                     "kv_free_positions", "kv_reserve_positions", "kv_ok")},
        # chaleur du contexte de la session qui demande (informatif : ne change pas
        # `free`, a ce stade — voir warmth_view)
        "warmth": warmth_view(session, tokens, signals, now),
        "lease": lease,
        "pause": pause_view(now),
        "server_time": round(now, 3),
    }


# ---------------------------------------------------------------------------- bail

def acquire(agent: str, task: str, ttl: int) -> tuple[int, dict]:
    now = time.time()
    ttl = max(60, min(int(ttl or CONFIG["lease_ttl_s"]), CONFIG["lease_max_ttl_s"]))
    with _LOCK:
        lease = lease_active(now)
        if lease and lease.get("agent") == agent:
            lease["expires_at"] = now + ttl
            lease["task"] = task or lease.get("task")
            lease["renewals"] = int(lease.get("renewals", 0)) + 1
            _persist(force=True)
            log("lease_refreshed", agent=agent, task=lease.get("task"), ttl=ttl)
            return 200, dict(lease)
        if lease:
            remaining = round(float(lease["expires_at"]) - now, 1)
            log("lease_denied", agent=agent, task=task, held_by=lease.get("agent"))
            return 409, {"error": "lease_held", "held_by": lease.get("agent"),
                         "task": lease.get("task"), "remaining_s": remaining,
                         "expires_at": lease.get("expires_at")}
        lease = {"id": "%s-%d-%04x" % (re.sub(r"[^a-z0-9]+", "-", agent.lower())[:24],
                                        int(now), os.getpid() & 0xFFF),
                 "agent": agent, "task": task, "acquired_at": round(now, 3),
                 "ttl": ttl, "expires_at": round(now + ttl, 3), "renewals": 0}
        STATE["lease"] = lease
        _persist(force=True)
        log("lease_granted", agent=agent, task=task, ttl=ttl, lease_id=lease["id"])
        return 200, dict(lease)


def renew(lease_id: str, ttl: int) -> tuple[int, dict]:
    now = time.time()
    ttl = max(60, min(int(ttl or CONFIG["lease_ttl_s"]), CONFIG["lease_max_ttl_s"]))
    with _LOCK:
        lease = lease_active(now)
        if not lease or lease.get("id") != lease_id:
            return 410, {"error": "lease_gone", "lease_id": lease_id}
        lease["expires_at"] = now + ttl
        lease["renewals"] = int(lease.get("renewals", 0)) + 1
        _persist(force=True)
        return 200, dict(lease)


def release(lease_id: str) -> tuple[int, dict]:
    now = time.time()
    with _LOCK:
        lease = lease_active(now)
        if not lease:
            stale = STATE.get("lease") or {}
            STATE["lease"] = None
            _persist(force=True)
            return 200, {"released": False, "note": "no active lease", "was": stale.get("id")}
        if lease.get("id") != lease_id:
            return 410, {"error": "lease_mismatch", "expected": lease.get("id"), "got": lease_id}
        STATE["lease"] = None
        STATE["last_activity_ts"] = now  # un job qui se termine relance le cooldown
        STATE["last_activity_reason"] = "lease_released"
        _persist(force=True)
        log("lease_released", agent=lease.get("agent"), task=lease.get("task"),
            lease_id=lease_id)
        return 200, {"released": True, "lease_id": lease_id, "agent": lease.get("agent")}


# ------------------------------------------------------------------- pause globale

def pause_active(now: float | None = None) -> dict | None:
    """Pause globale en cours, ou `None`. `until` null = indefinite ; une pause dont
    l'`until` est passe est consideree comme levee (elle reste dans l'etat jusqu'au
    prochain /pause ou /resume, mais ne bloque plus rien)."""
    pause = STATE.get("pause")
    if not isinstance(pause, dict):
        return None
    until = pause.get("until")
    if until is not None:
        try:
            until = float(until)
        except (TypeError, ValueError):
            return None
        if until <= (now or time.time()):
            return None
    return pause


def pause_view(now: float | None = None) -> dict:
    """Representation publique de la pause (GET /pause, cle « pause » de /free)."""
    now = now or time.time()
    pause = pause_active(now)
    if not pause:
        return {"paused": False, "until": None, "reason": "", "remaining_s": None}
    until = pause.get("until")
    until = float(until) if until is not None else None
    return {"paused": True, "until": until,
            "reason": str(pause.get("reason") or ""),
            "remaining_s": round(until - now, 1) if until is not None else None}


def set_pause(until, reason: str = "") -> tuple[int, dict]:
    """Pose la pause globale. `until` : epoch (float) ou `None` = indefinite."""
    now = time.time()
    until_ts = None
    if until is not None:
        try:
            until_ts = float(until)
        except (TypeError, ValueError):
            return 400, {"error": "until must be a number (epoch seconds) or null"}
        if not math.isfinite(until_ts):
            return 400, {"error": "until must be a finite epoch timestamp or null"}
    with _LOCK:
        STATE["pause"] = {"until": until_ts, "reason": str(reason or "")[:400],
                          "set_at": round(now, 3)}
        _persist(force=True)
        log("pause_set", until=until_ts, reason=str(reason or "")[:200],
            remaining_s=None if until_ts is None else round(until_ts - now, 1))
        return 200, pause_view(now)


def clear_pause() -> tuple[int, dict]:
    """Leve la pause globale (meme si elle a deja expire)."""
    now = time.time()
    with _LOCK:
        was = STATE.get("pause") if isinstance(STATE.get("pause"), dict) else None
        STATE["pause"] = None
        _persist(force=True)
        log("pause_cleared", was_until=(was or {}).get("until"),
            was_reason=(was or {}).get("reason"))
        return 200, pause_view(now)


# ------------------------------------------------------------------- chaleur de cache

def _warmth_prune(now: float) -> dict:
    """Jette les entrées périmées (instance morte = ne doit pas protéger du vide) et
    plafonne la table. Retourne la table vivante."""
    table = STATE.get("warmth")
    if not isinstance(table, dict):
        table = {}
        STATE["warmth"] = table
    live = {sid: row for sid, row in table.items()
            if isinstance(row, dict) and now - float(row.get("ts", 0)) <= CONFIG["warm_ttl_s"]}
    if len(live) > CONFIG["warm_max_sessions"]:
        # on garde les plus récents : les autres sont froids par construction
        live = dict(sorted(live.items(), key=lambda kv: -float(kv[1].get("ts", 0)))
                    [:CONFIG["warm_max_sessions"]])
    STATE["warmth"] = live
    return live


def report_warmth(session: str, prompt_tokens, cached_tokens, tag: str = "",
                  evicted_total=None, agent: str = "", now: float | None = None) -> tuple[int, dict]:
    """Ce qu'une instance Hermes sait de SON dernier appel — et c'est tout ce que
    l'arbitre peut savoir de son contexte : le moteur ne publie aucune clé d'entrée.

    `cached_tokens` absent (appel froid : le moteur ne renvoie alors pas la clé du
    tout) compte pour 0, c'est exactement le signal recherché.
    """
    now = now or time.time()
    sid = str(session or "").strip()
    if not sid:
        return 400, {"error": "session is required"}
    try:
        prompt_tokens = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        return 400, {"error": "prompt_tokens must be an integer"}
    if prompt_tokens < 0:
        return 400, {"error": "prompt_tokens must be >= 0"}
    try:
        cached_tokens = int(cached_tokens) if cached_tokens is not None else 0
    except (TypeError, ValueError):
        cached_tokens = 0
    cached_tokens = max(0, min(cached_tokens, prompt_tokens)) if prompt_tokens else 0
    with _LOCK:
        _warmth_prune(now)
        STATE["warmth"][sid] = {
            "ts": round(now, 3), "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "ratio": round(cached_tokens / prompt_tokens, 4) if prompt_tokens else None,
            "tag": str(tag or "")[:64], "agent": str(agent or "")[:40],
            "evicted_at_report": None if evicted_total is None else float(evicted_total),
        }
        _persist()
    # le rapport porte sa propre valeur du compteur d'evictions : `evicted_since`
    # vaut alors 0, et une vue sans signal moteur ne reste pas muette sur ce point.
    return 200, warmth_view(sid, None, {"cache_evicted_total": evicted_total}, now)


def warmth_view(session: str, tokens=None, signals: dict | None = None,
                now: float | None = None) -> dict:
    """Le verdict de chaleur — **informatif à ce stade**, il ne change pas `free`.

    Chaud = dernier ratio de cache bon + vu récemment + personne n'a évidé le cache
    depuis. Le troisième terme est ce qui rend le dispositif correct avec plusieurs
    instances Hermes : elles s'ignorent, mais elles lisent toutes le même compteur
    d'évictions du moteur.

    `reclaimable_tokens` répond à « est-ce qu'il y a du froid à évincer » : l'empreinte
    des sessions vues il y a plus de `protect_age_s`. En dessous de cet âge on considère
    que la session est en cours et on ne compte pas sur son cache.

    `advice` est la réponse brute à « on y va ou pas ? » : `go` (contexte résident, rien
    à recharger), `cold-room` (à froid, mais de la place évictable), `cold-no-room`
    (à froid et tout le cache appartient à des sessions en cours), `cold` (taille non
    déclarée, pas de verdict de place).
    """
    now = now or time.time()
    signals = signals or {}
    with _LOCK:
        table = _warmth_prune(now)
        row = table.get(session) if session else None
        others = {sid: r for sid, r in table.items() if sid != session}
    grace = CONFIG["warm_grace_s"]
    protect = CONFIG["protect_age_s"]
    protected = sum(int(r.get("prompt_tokens") or 0) for r in others.values()
                    if now - float(r.get("ts", 0)) < protect)
    reclaimable = sum(int(r.get("prompt_tokens") or 0) for r in others.values()
                      if now - float(r.get("ts", 0)) >= protect)
    evicted_now = signals.get("cache_evicted_total")
    out = {"known": bool(row), "warm": False, "reason": "unknown", "age_s": None,
           "ratio": None, "prompt_tokens": None, "evicted_since": None,
           "protected_tokens": protected, "reclaimable_tokens": reclaimable,
           "grace_s": grace, "protect_age_s": protect}
    if tokens is not None:
        try:
            out["needs_tokens"] = int(tokens)
        except (TypeError, ValueError):
            out["needs_tokens"] = None
    if row:
        age = max(0.0, now - float(row.get("ts", 0)))
        ratio = row.get("ratio")
        out.update({"age_s": round(age, 1), "ratio": ratio,
                    "prompt_tokens": row.get("prompt_tokens")})
        if evicted_now is not None and row.get("evicted_at_report") is not None:
            out["evicted_since"] = max(0, int(evicted_now - float(row["evicted_at_report"])))
        if age > grace:
            out["reason"] = "stale"
        elif ratio is None or ratio < CONFIG["warm_ratio_min"]:
            out["reason"] = "cold"
        elif (out["evicted_since"] or 0) > 0:
            out["reason"] = "evicted"
        else:
            out["warm"] = True
            out["reason"] = "hot"
    # Verdict de place. Une session chaude n'a rien a recharger : sa place est deja
    # prise, elle n'a besoin de rien evincer. Une session froide, elle, ne passe que
    # dans l'empreinte des sessions froides — jamais dans celle d'une session qui
    # tourne (protected_tokens).
    if out.get("needs_tokens") is not None:
        out["room_for_cold"] = bool(out["warm"] or reclaimable >= out["needs_tokens"])
    # La reponse a « on y va ou pas ? ». Informatif : `free` ne le regarde pas encore.
    out["advice"] = ("go" if out["warm"]
                     else "cold-room" if out.get("room_for_cold")
                     else "cold-no-room" if out.get("room_for_cold") is False
                     else "cold")
    return out


# --------------------------------------------------------------------- HTTP handler

class Handler(BaseHTTPRequestHandler):
    server_version = "ia-free/1.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload, content_type="application/json") -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8", "replace"))
        except ValueError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        query = urllib.parse.parse_qs(self.path.split("?", 1)[1] if "?" in self.path else "")
        if path in ("/free", "/status"):
            # `?session=<id>&tokens=<n>` : la session declare pour qui on demande, et
            # la taille de contexte qu'elle va charger. Sans parametre, `warmth` est
            # quand meme present (inconnu + totaux) — reponse retro-compatible.
            verdict = evaluate(session=(query.get("session") or [""])[0],
                               tokens=(query.get("tokens") or [None])[0])
            if path == "/status":
                verdict["config"] = {k: CONFIG[k] for k in
                                     ("metrics_url", "slots_url", "gpu_max", "max_inflight",
                                      "cooldown_s", "lease_ttl_s", "lease_max_ttl_s",
                                      "idle_sample_s", "idle_min_s", "warm_grace_s",
                                      "warm_ratio_min", "protect_age_s", "warm_ttl_s")}
                verdict["uptime_s"] = round(time.time() - STATE["started_at"], 1)
                verdict["engine_epoch"] = STATE.get("engine_epoch")
                verdict["lease"] = lease_active()
            self._send(200, verdict)
        elif path == "/warmth":
            with _LOCK:
                table = dict(STATE.get("warmth") or {})
            self._send(200, {"sessions": len(table), "warmth": table})
        elif path == "/lease":
            self._send(200, {"lease": lease_active()})
        elif path == "/pause":
            self._send(200, pause_view())
        elif path == "/history":
            self._send(200, {"events": list(HISTORY)})
        elif path in ("/healthz", "/health"):
            self._send(200, b"ok\n", "text/plain")
        else:
            self._send(404, {"error": "unknown path", "paths":
                             ["/free", "/status", "/pause", "/resume", "/lease",
                              "/lease/renew", "/warmth", "/history", "/healthz"]})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        body = self._json_body()
        if path == "/lease":
            agent = str(body.get("agent") or "").strip()
            if not agent:
                self._send(400, {"error": "agent is required"})
                return
            code, payload = acquire(agent, str(body.get("task") or "")[:400],
                                    int(body.get("ttl") or CONFIG["lease_ttl_s"]))
            self._send(code, payload)
        elif path == "/lease/renew":
            code, payload = renew(str(body.get("lease_id") or ""),
                                  int(body.get("ttl") or CONFIG["lease_ttl_s"]))
            self._send(code, payload)
        elif path == "/pause":
            # {"until": epoch||null, "reason": str} — les deux champs sont optionnels :
            # corps vide ou `until` absent/null = pause indefinite.
            code, payload = set_pause(body.get("until"), str(body.get("reason") or ""))
            self._send(code, payload)
        elif path == "/resume":
            code, payload = clear_pause()
            self._send(code, payload)
        elif path == "/warmth":
            # {"session":str, "prompt_tokens":int, "cached_tokens":int|null,
            #  "tag":str?, "evicted_total":number?, "agent":str?}
            # cached_tokens null = le moteur n'a rien rendu (appel froid) => 0.
            code, payload = report_warmth(
                str(body.get("session") or ""), body.get("prompt_tokens"),
                body.get("cached_tokens"),
                str(body.get("tag") or body.get("prefix") or ""),
                body.get("evicted_total"), str(body.get("agent") or ""))
            self._send(code, payload)
        else:
            self._send(404, {"error": "unknown path"})

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path.startswith("/lease/"):
            code, payload = release(path[len("/lease/"):])
            self._send(code, payload)
        elif path == "/lease":
            lease = lease_active()
            if not lease:
                self._send(200, {"released": False, "note": "no active lease"})
            else:
                code, payload = release(lease["id"])
                self._send(code, payload)
        else:
            self._send(404, {"error": "use DELETE /lease/<id>"})

    def log_message(self, fmt, *args) -> None:  # journal compact sur stdout
        if self.path.split("?")[0] not in ("/free", "/healthz", "/health"):
            log("http", method=self.command, path=self.path,
                src=self.client_address[0] if self.client_address else "")


# ----------------------------------------------------------------------------- main

def probe(url: str | None, timeout: float = 5.0) -> int:
    """Mode client : interroge un serveur (ou l'evaluation locale) et rend le verdict."""
    if url:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            verdict = json.loads(resp.read().decode())
    else:
        verdict = evaluate()
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0 if verdict.get("free") else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", nargs="?", const="", default=None, metavar="URL",
                        help="mode client : affiche le verdict et quitte (code 0 si libre)")
    parser.add_argument("--once", action="store_true", help="evalue localement une fois et quitte")
    parser.add_argument("--port", type=int, help="couvre IA_FREE_PORT")
    parser.add_argument("--bind", help="couvre IA_FREE_BIND")
    args = parser.parse_args(argv)

    if args.probe is not None:
        return probe(args.probe or ("http://127.0.0.1:%d/free" % (args.port or CONFIG["port"])))
    if args.once:
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
        return 0

    if args.port:
        CONFIG["port"] = args.port
    if args.bind:
        CONFIG["bind"] = args.bind
    _restore()
    idle_stop = threading.Event()
    if CONFIG["idle_sample_s"] > 0:
        threading.Thread(target=idle_sampler, args=(idle_stop,), name="ia-free-idle",
                         daemon=True).start()
    server = ThreadingHTTPServer((CONFIG["bind"], CONFIG["port"]), Handler)
    server.daemon_threads = True
    log("start", bind=CONFIG["bind"], port=CONFIG["port"],
        metrics_url=CONFIG["metrics_url"], gpu_max=CONFIG["gpu_max"],
        cooldown_s=CONFIG["cooldown_s"], lease_ttl_s=CONFIG["lease_ttl_s"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        idle_stop.set()
        _persist(force=True)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
