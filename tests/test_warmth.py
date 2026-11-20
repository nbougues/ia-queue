#!/usr/bin/env python3
"""Tests du registre de chaleur de cache (ia-free-server).

    python3 -m unittest discover -s tests          # ou : python3 tests/test_warmth.py

Le moteur d'inférence est simulé par un faux `_http_get` (texte Prometheus en dur) ;
l'état persiste dans un répertoire temporaire. Aucun réseau, aucun GPU, aucun service
touché.

Ce que ces tests verrouillent :
  * la chaleur se DÉDUIT (ratio du dernier appel + récence + compteur d'évictions du
    moteur), elle ne se demande pas au moteur — il ne publie aucune clé d'entrée ;
  * le compteur d'évictions est le seul signal qui fonctionne entre plusieurs
    instances Hermes qui s'ignorent ;
  * `warmth` est INFORMATIF : il n'affecte pas `free` (phase d'observation) ;
  * `protected` / `reclaimable` : ce qui est « en cours » n'est pas compté comme
    évictable.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# a faire AVANT l'import : CONFIG est fige a l'import
_TMP = Path(tempfile.mkdtemp(prefix="ia-free-warmth-"))
os.environ["IA_FREE_STATE_FILE"] = str(_TMP / "state.json")
os.environ["IA_FREE_IDLE_SAMPLE_S"] = "0"
os.environ["IA_FREE_WARM_GRACE_S"] = "90"
os.environ["IA_FREE_PROTECT_AGE_S"] = "60"
os.environ["IA_FREE_WARM_RATIO_MIN"] = "0.5"
os.environ["IA_FREE_WARM_TTL_S"] = "900"
os.environ["IA_FREE_WARM_MAX_SESSIONS"] = "5"

import ia_free_server as srv  # noqa: E402

_LOG_SINK = (_TMP / "service.log").open("w", encoding="utf-8")
_LOG_LOCK = threading.Lock()
_real_log = srv.log


def _silent_log(event, **fields):
    with _LOG_LOCK, redirect_stdout(_LOG_SINK):
        _real_log(event, **fields)


srv.log = _silent_log

# ------------------------------------------------------------------ moteur simulé
# Valeurs réelles relevées sur le moteur (inference-host:8731) : le ratio correct
# cached/(cached+prompt) doit reproduire le `token_hit_rate` que /cache publie.
CACHED = 3_209_839_450
PROMPT_TRAITED = 304_187_021
ENGINE_TOKEN_HIT_RATE = 0.9134          # lu sur GET /cache du moteur réel

METRICS = """llamacpp:requests_processing 0
llamacpp:requests_deferred 0
llamacpp:kv_cache_usage_ratio 0.99
llamacpp:prompt_tokens_total %d
llamacpp:kv_cache_tokens 780544
llamacpp:predicted_tokens_seconds 38.0
llamacpp:prompt_tokens_seconds 1286.0
halogen:requests_total 39816
halogen:busy_for_seconds 0
halogen:prompt_tokens_cached_total %d
halogen:cache_evicted_total 44410
halogen:cache_hits_total 36571
halogen:cache_misses_total 3408
halogen:cache_entries 24
halogen:kv_pool_positions 1572864
""" % (PROMPT_TRAITED, CACHED)


def metrics_with(evicted: int, cached: int = CACHED) -> str:
    return METRICS.replace("halogen:cache_evicted_total 44410",
                           "halogen:cache_evicted_total %d" % evicted).replace(
        "halogen:prompt_tokens_cached_total %d" % CACHED,
        "halogen:prompt_tokens_cached_total %d" % cached)


def fake_http_get(url: str) -> str:
    if url.endswith("/slots"):
        return "[]"
    if url.endswith("/metrics"):
        return getattr(fake_http_get, "metrics", METRICS)
    raise urllib.error.URLError("test: no such url %s" % url)


class WarmthCase(unittest.TestCase):
    # les seuils sont poses sur CONFIG, pas sur l'environnement : quand `discover`
    # execute test_pause.py en premier, c'est son import qui a fige CONFIG, et les
    # variables d'environnement posees ici n'auraient aucun effet.
    KNOBS = {"warm_grace_s": 90.0, "warm_ratio_min": 0.5, "protect_age_s": 60.0,
             "warm_ttl_s": 900.0, "warm_max_sessions": 5}

    def setUp(self):
        self._real_get = srv._http_get
        srv._http_get = fake_http_get
        fake_http_get.metrics = METRICS
        srv.CONFIG["state_file"] = str(_TMP / ("state-%s.json" % self.id()))
        self._saved_config = {k: srv.CONFIG[k] for k in self.KNOBS}
        srv.CONFIG.update(self.KNOBS)
        with srv._LOCK:
            srv.STATE.update({"lease": None, "pause": None, "warmth": {},
                              "engine_epoch": 0,
                              "last_activity_ts": time.time(),
                              "last_busy_ts": time.time(),
                              "requests_total_seen": None})

    def tearDown(self):
        srv.CONFIG.update(self._saved_config)
        srv._http_get = self._real_get

    def quiet(self, fn, *a, **kw):
        with redirect_stdout(io.StringIO()):
            return fn(*a, **kw)

    def report(self, session, prompt, cached, evicted=44410, agent="t"):
        code, view = self.quiet(srv.report_warmth, session, prompt, cached,
                                "prefix-of-%s" % session, evicted, agent)
        self.assertEqual(code, 200)
        return view

    def age(self, session, seconds):
        """Recale une entrée dans le passé (vieillissement simulé)."""
        srv.STATE["warmth"][session]["ts"] = time.time() - seconds


class TestDeduction(WarmthCase):
    """La chaleur se déduit ; les quatre états doivent être distingués."""

    def test_hot(self):
        view = self.report("s1", 30000, 29400)
        self.assertTrue(view["warm"])
        self.assertEqual(view["reason"], "hot")
        self.assertEqual(view["ratio"], 0.98)
        self.assertEqual(view["prompt_tokens"], 30000)
        self.assertEqual(view["evicted_since"], 0)
        self.assertEqual(view["advice"], "go")

    def test_cold_when_the_engine_returned_no_cached_tokens(self):
        # appel froid : le moteur ne renvoie AUCUNE clé cached_tokens -> None -> 0
        view = self.report("s1", 30000, None)
        self.assertFalse(view["warm"])
        self.assertEqual(view["reason"], "cold")
        self.assertEqual(view["ratio"], 0.0)

    def test_unknown_session(self):
        view = self.quiet(srv.warmth_view, "jamais-vue")
        self.assertFalse(view["known"])
        self.assertEqual(view["reason"], "unknown")
        self.assertFalse(view["warm"])

    def test_stale_after_the_grace_period(self):
        self.report("s1", 30000, 29400)
        self.age("s1", 91)
        view = self.quiet(srv.warmth_view, "s1")
        self.assertEqual(view["reason"], "stale")
        self.assertFalse(view["warm"])

    def test_evicted_by_another_instance(self):
        """Le cas multi-instances : rien n'avertit s1, mais le compteur du moteur, lui,
        est lu par tout le monde."""
        self.report("s1", 30000, 29400, evicted=44410)
        fake_http_get.metrics = metrics_with(44413)      # une autre instance a évidé
        view = self.quiet(srv.warmth_view, "s1", None, {"cache_evicted_total": 44413})
        self.assertEqual(view["evicted_since"], 3)
        self.assertEqual(view["reason"], "evicted")
        self.assertFalse(view["warm"])

    def test_low_ratio_beats_eviction_in_reporting_order(self):
        self.report("s1", 30000, 100, evicted=44410)
        view = self.quiet(srv.warmth_view, "s1", None, {"cache_evicted_total": 44499})
        self.assertEqual(view["reason"], "cold")


class TestProtectedFootprint(WarmthCase):
    """« Est-ce qu'il y a du froid à évincer » : ce qui est récent ne compte pas."""

    def test_split_between_young_and_old(self):
        self.report("jeune", 10000, 9000)
        self.report("vieux", 20000, 19000)
        self.age("vieux", 61)
        view = self.quiet(srv.warmth_view, "moi", 15000)
        self.assertEqual(view["protected_tokens"], 10000)
        self.assertEqual(view["reclaimable_tokens"], 20000)
        self.assertTrue(view["room_for_cold"])          # 20000 >= 15000
        self.assertEqual(view["needs_tokens"], 15000)
        self.assertEqual(view["advice"], "cold-room")

    def test_no_room_when_everyone_is_running(self):
        self.report("jeune1", 10000, 9000)
        self.report("jeune2", 10000, 9000)
        view = self.quiet(srv.warmth_view, "moi", 15000)
        self.assertEqual(view["reclaimable_tokens"], 0)
        self.assertEqual(view["protected_tokens"], 20000)
        self.assertFalse(view["room_for_cold"])
        self.assertEqual(view["advice"], "cold-no-room")

    def test_a_warm_session_needs_nothing_evicted(self):
        """Meme si le pool est plein de sessions recentes : son contexte est deja la."""
        self.report("jeune", 10000, 9000)
        self.report("s1", 30000, 29700)
        view = self.quiet(srv.warmth_view, "s1", 30000)
        self.assertEqual(view["reclaimable_tokens"], 0)
        self.assertTrue(view["room_for_cold"])
        self.assertEqual(view["advice"], "go")

    def test_advice_without_a_declared_size(self):
        self.report("s1", 30000, None)
        self.assertEqual(self.quiet(srv.warmth_view, "s1")["advice"], "cold")

    def test_self_is_not_its_own_reclaimable(self):
        self.report("s1", 30000, 29400)
        self.age("s1", 500)
        view = self.quiet(srv.warmth_view, "s1", 1000)
        self.assertEqual(view["reclaimable_tokens"], 0)

    def test_no_tokens_param_means_no_room_verdict(self):
        view = self.quiet(srv.warmth_view, "s1")
        self.assertNotIn("needs_tokens", view)
        self.assertNotIn("room_for_cold", view)


class TestHygiene(WarmthCase):
    def test_ttl_drops_dead_instances(self):
        self.report("mort", 30000, 29000)
        self.age("mort", 901)
        view = self.quiet(srv.warmth_view, "vivant")
        self.assertEqual(view["protected_tokens"], 0)
        self.assertEqual(view["reclaimable_tokens"], 0)
        self.assertNotIn("mort", srv.STATE["warmth"])

    def test_table_is_capped_keeping_the_newest(self):
        for i in range(8):
            self.report("s%d" % i, 1000, 900)
            srv.STATE["warmth"]["s%d" % i]["ts"] = time.time() - (8 - i)
        self.quiet(srv.warmth_view, "x")
        self.assertEqual(len(srv.STATE["warmth"]), 5)
        self.assertIn("s7", srv.STATE["warmth"])
        self.assertNotIn("s0", srv.STATE["warmth"])

    def test_validation(self):
        self.assertEqual(self.quiet(srv.report_warmth, "", 10, 5)[0], 400)
        self.assertEqual(self.quiet(srv.report_warmth, "s", "abc", 5)[0], 400)
        self.assertEqual(self.quiet(srv.report_warmth, "s", -1, 5)[0], 400)

    def test_cached_never_exceeds_prompt(self):
        view = self.report("s1", 1000, 99999)
        self.assertEqual(view["ratio"], 1.0)

    def test_zero_prompt_tokens(self):
        view = self.report("s1", 0, 0)
        self.assertIsNone(view["ratio"])
        self.assertEqual(view["reason"], "cold")

    def test_persistence_round_trip(self):
        self.report("s1", 30000, 29400)
        self.quiet(srv._persist, True)
        srv.STATE["warmth"] = {}
        self.assertFalse(self.quiet(srv.warmth_view, "s1")["known"])
        self.quiet(srv._restore)
        self.assertTrue(self.quiet(srv.warmth_view, "s1")["warm"])


class TestCacheHitRatio(WarmthCase):
    """Le ratio publié doit reproduire celui du moteur, pas un rapport farfelu."""

    def test_matches_the_engine_own_token_hit_rate(self):
        sig = self.quiet(srv.read_engine)
        self.assertAlmostEqual(sig["cache_hit_ratio"], ENGINE_TOKEN_HIT_RATE, places=4)

    def test_cache_health_signals_are_exposed(self):
        sig = self.quiet(srv.read_engine)
        self.assertEqual(sig["cache_evicted_total"], 44410)
        self.assertEqual(sig["cache_entries"], 24)
        self.assertEqual(sig["kv_pool_positions"], 1572864)
        self.assertEqual(sig["kv_cache_tokens"], 780544)


class TestVerdictUnchanged(WarmthCase):
    """Phase d'observation : warmth s'AJOUTE, il ne décrète rien."""

    def test_free_is_untouched_by_warmth(self):
        self.report("s1", 30000, 29400)
        verdict = self.quiet(srv.evaluate, "s1", 30000)
        self.assertTrue(verdict["free"], "la chaleur ne doit pas bloquer")
        self.assertEqual(verdict["reason"], "free")
        self.assertTrue(verdict["warmth"]["warm"])

    def test_a_cold_session_is_not_blocked_yet(self):
        self.report("s1", 30000, None)
        verdict = self.quiet(srv.evaluate, "s1", 30000)
        self.assertTrue(verdict["free"])
        self.assertEqual(verdict["warmth"]["reason"], "cold")

    def test_warmth_key_is_always_present(self):
        verdict = self.quiet(srv.evaluate)
        self.assertIn("warmth", verdict)
        self.assertEqual(verdict["warmth"]["reason"], "unknown")

    def test_pause_still_wins(self):
        self.report("s1", 30000, 29400)
        self.quiet(srv.set_pause, None, "maintenance")
        verdict = self.quiet(srv.evaluate, "s1", 30000)
        self.assertFalse(verdict["free"])
        self.assertEqual(verdict["reason"], "paused")


class TestEngineRestart(WarmthCase):
    """Un redémarrage du moteur remet tous ses compteurs à zéro et vide son cache KV.

    C'est le cas qui rend le delta d'évictions dangereux : un compteur qui a
    décrémenté donne `max(0, …) = 0`, donc « personne n'a évidé », donc une session
    déclarée chaude juste avant le redémarrage resterait **chaude** alors que le
    moteur a tout perdu — exactement le faux positif qui enverrait une session froide
    croire qu'elle n'a rien à recharger.
    """

    def metrics(self, requests_total: int, evicted: int = 44410) -> str:
        return METRICS.replace("halogen:requests_total 39816",
                               "halogen:requests_total %d" % requests_total).replace(
            "halogen:cache_evicted_total 44410",
            "halogen:cache_evicted_total %d" % evicted)

    def test_restart_wipes_the_warmth_table(self):
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.report("s1", 30000, 29400)
        self.assertTrue(self.quiet(srv.warmth_view, "s1")["warm"])
        fake_http_get.metrics = self.metrics(2, evicted=0)      # moteur tout juste relancé
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.assertEqual(srv.STATE["engine_epoch"], 1)
        self.assertEqual(srv.STATE["warmth"], {})
        view = self.quiet(srv.warmth_view, "s1")
        self.assertFalse(view["known"])
        self.assertFalse(view["warm"])
        self.assertEqual(view["reason"], "unknown")

    def test_epoch_survives_a_persist_round_trip(self):
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.report("s1", 30000, 29400)
        fake_http_get.metrics = self.metrics(1, evicted=0)
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.quiet(srv._persist, True)
        srv.STATE["engine_epoch"], srv.STATE["warmth"] = 0, {}
        self.quiet(srv._restore)
        self.assertEqual(srv.STATE["engine_epoch"], 1)

    def test_normal_growth_is_not_a_restart(self):
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.report("s1", 30000, 29400)
        fake_http_get.metrics = self.metrics(39817)
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.assertEqual(srv.STATE["engine_epoch"], 0)
        self.assertTrue(self.quiet(srv.warmth_view, "s1")["warm"])

    def test_restart_detected_without_any_baseline(self):
        """Le cas reel de l'upgrade : l'arbitre redemarre APRES le moteur. Aucun
        `requests_total_seen` en memoire, rien a comparer — seul le compteur
        d'evictions porte la preuve, et il la porte."""
        self.report("s1", 30000, 29400, evicted=44469)     # herite d'avant l'upgrade
        srv.STATE["requests_total_seen"] = None            # arbitre fraichement relance
        fake_http_get.metrics = self.metrics(17, evicted=0)
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.assertEqual(srv.STATE["engine_epoch"], 1)
        self.assertEqual(srv.STATE["warmth"], {})
        self.assertFalse(self.quiet(srv.warmth_view, "s1")["warm"])

    def test_a_cold_engine_with_nothing_inherited_is_not_a_restart(self):
        fake_http_get.metrics = self.metrics(17, evicted=0)
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.assertEqual(srv.STATE["engine_epoch"], 0)

    def test_counters_are_part_of_the_persisted_state(self):
        self.report("s1", 30000, 29400)
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.quiet(srv._persist, True)
        srv.STATE["requests_total_seen"] = None
        self.quiet(srv._restore)
        self.assertEqual(srv.STATE["requests_total_seen"], 39816)

    def test_a_second_restart_counts_again(self):
        self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        for total in (3, 1):
            fake_http_get.metrics = self.metrics(total, evicted=0)
            self.quiet(srv.track_idle, self.quiet(srv.read_engine), time.time())
        self.assertEqual(srv.STATE["engine_epoch"], 2)   # 39816 -> 3, puis 3 -> 1


class TestHttpSurface(WarmthCase):
    """Le vrai Handler, sur un port libre : ce que voient les instances Hermes."""

    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        cls.httpd.daemon_threads = True
        cls.base = "http://127.0.0.1:%d" % cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as resp:
            return json.loads(resp.read().decode())

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode())

    def test_report_then_free_carries_the_view(self):
        code, view = self.post("/warmth", {"session": "http1", "prompt_tokens": 30000,
                                           "cached_tokens": 29700, "agent": "agent-host"})
        self.assertEqual(code, 200)
        self.assertTrue(view["warm"])
        verdict = self.get("/free?session=http1&tokens=30000")
        self.assertEqual(verdict["warmth"]["reason"], "hot")
        self.assertEqual(verdict["warmth"]["needs_tokens"], 30000)
        self.assertIn("cache_evicted_total", verdict["signals"])

    def test_free_without_params_stays_backward_compatible(self):
        verdict = self.get("/free")
        for key in ("free", "reason", "calm_for_s", "idle_for_s", "idle_ok",
                    "signals", "lease", "pause"):
            self.assertIn(key, verdict)
        self.assertEqual(verdict["warmth"]["reason"], "unknown")

    def test_get_warmth_lists_the_table(self):
        self.post("/warmth", {"session": "http2", "prompt_tokens": 1000,
                              "cached_tokens": 900})
        body = self.get("/warmth")
        self.assertIn("http2", body["warmth"])
        self.assertGreaterEqual(body["sessions"], 1)

    def test_post_warmth_rejects_garbage(self):
        code, body = self.post("/warmth", {"prompt_tokens": 10})
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_status_exposes_the_knobs(self):
        cfg = self.get("/status")["config"]
        for key in ("warm_grace_s", "warm_ratio_min", "protect_age_s", "warm_ttl_s"):
            self.assertIn(key, cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
