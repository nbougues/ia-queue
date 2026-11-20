#!/usr/bin/env python3
"""Tests de la pause globale et des signaux de performance de ia-free-server.

    python3 -m unittest discover -s tests -v         # ou : python3 tests/test_pause.py

Aucun reseau, aucun GPU, aucun service touche : le moteur d'inférence est simule par
un faux `_http_get` (texte Prometheus en dur) et le vrai `Handler` est ecoute sur un
port libre de 127.0.0.1. L'etat persiste est ecrit dans un repertoire temporaire.
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
_TMP = Path(tempfile.mkdtemp(prefix="ia-free-test-"))
os.environ["IA_FREE_STATE_FILE"] = str(_TMP / "state.json")
os.environ["IA_FREE_IDLE_SAMPLE_S"] = "0"   # pas de sondeur d'oisivete pendant les tests

import ia_free_server as srv  # noqa: E402

# Le service journalise chaque evenement sur stdout (c'est son contrat systemd) ; on
# dérive ce journal vers un fichier du tmpdir pour garder une sortie de test lisible,
# sans toucher a la logique `log()` (HISTORY reste alimente, /history reste testable).
_LOG_SINK = (_TMP / "service.log").open("w", encoding="utf-8")
_LOG_LOCK = threading.Lock()
_real_log = srv.log


def _silent_log(event, **fields):
    with _LOG_LOCK, redirect_stdout(_LOG_SINK):
        _real_log(event, **fields)


srv.log = _silent_log

# --------------------------------------------------------------------- moteur simule

METRICS_FULL = """# HELP llamacpp:requests_processing Inference requests in flight
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 0
llamacpp:requests_deferred 0
llamacpp:kv_cache_usage_ratio 0.02
llamacpp:prompt_tokens_total 100000
llamacpp:prompt_seconds_total 77.9
llamacpp:tokens_predicted_total 4000
llamacpp:tokens_predicted_seconds_total 105.2
llamacpp:prompt_tokens_seconds 1286.0
llamacpp:predicted_tokens_seconds 38.0
llamacpp:kv_cache_tokens 512
halogen:busy_for_seconds 0
halogen:requests_total 41
halogen:prompt_tokens_cached_total 62000
halogen:queue_wait_seconds_total 12.5
halogen:queue_wait_longest_seconds 3.25
halogen:kv_pool_reserved_tokens 0
"""

# seules les series « liberte » : tout le reste est absent (moteur plus ancien)
METRICS_BARE = """llamacpp:requests_processing 0
llamacpp:requests_deferred 0
halogen:requests_total 7
"""

METRICS_CACHE_OVERFLOW = METRICS_FULL.replace("halogen:prompt_tokens_cached_total 62000",
                                              "halogen:prompt_tokens_cached_total 250000")
METRICS_ZERO_PROMPT_TOTAL = METRICS_FULL.replace("llamacpp:prompt_tokens_total 100000",
                                                 "llamacpp:prompt_tokens_total 0")
METRICS_LABELED = """llamacpp:requests_processing 0
llamacpp:predicted_tokens_seconds{model="qwen"} 40.5
halogen:prompt_tokens_cached_total{model="qwen"} 30
llamacpp:prompt_tokens_total{model="qwen"} 100
"""


def fake_http_get(url: str) -> str:
    if url.endswith("/slots"):
        return "[]"
    if url.endswith("/metrics"):
        return getattr(fake_http_get, "metrics", METRICS_FULL)
    raise urllib.error.URLError("test: no such url %s" % url)


def set_metrics(text: str) -> None:
    fake_http_get.metrics = text


# -------------------------------------------------------------------- harnais HTTP

FREE_KEYS = {"free", "reason", "calm_for_s", "idle_for_s", "idle_ok", "idle_min_s",
             "cooldown_s", "max_inflight", "signals", "lease"}
SIGNAL_KEYS = {"requests_processing", "requests_deferred", "inflight", "slots_total",
               "kv_cache_ratio", "gpu_busy_percent", "busy_for_seconds", "requests_total",
               "decode_tps", "prefill_tps", "cache_hit_ratio",
               "queue_wait_seconds_total", "queue_wait_longest_seconds"}
PAUSE_KEYS = {"paused", "until", "reason", "remaining_s"}


class EngineCase(unittest.TestCase):
    """Cas « fonction » : read_engine() / evaluate() sans serveur HTTP."""

    def setUp(self):
        self._real_get = srv._http_get
        srv._http_get = fake_http_get
        set_metrics(METRICS_FULL)
        self.state_file = Path(_TMP) / ("state-%s.json" % self.id())
        srv.CONFIG["state_file"] = str(self.state_file)
        with srv._LOCK:
            srv.STATE.update({"lease": None, "pause": None,
                              "last_activity_ts": time.time(),
                              "last_busy_ts": time.time(),
                              "requests_total_seen": None})

    def tearDown(self):
        srv._http_get = self._real_get

    def quiet(self, fn, *a, **kw):
        with redirect_stdout(io.StringIO()):
            return fn(*a, **kw)


class TestPerfSignals(EngineCase):

    def test_signals_read_from_single_scrape(self):
        set_metrics(METRICS_FULL)
        calls = []
        real = srv._http_get

        def counting(url):
            calls.append(url)
            return real(url)
        srv._http_get = counting
        sig = self.quiet(srv.read_engine)
        self.assertTrue(sig["reachable"])
        self.assertEqual(sig["decode_tps"], 38.0)
        self.assertEqual(sig["prefill_tps"], 1286.0)
        # 62 000 couverts / (62 000 couverts + 100 000 traites) : les deux compteurs
        # sont des parts d'un meme total, pas un rapport L'un-sur-L'autre.
        self.assertAlmostEqual(sig["cache_hit_ratio"], 62000 / 162000)
        self.assertEqual(sig["queue_wait_seconds_total"], 12.5)
        self.assertEqual(sig["queue_wait_longest_seconds"], 3.25)
        # un seul scrape /metrics : aucun appel supplementaire pour ces 5 signaux
        self.assertEqual(calls.count(srv.CONFIG["metrics_url"]), 1)

    def test_missing_metrics_are_none_not_errors(self):
        set_metrics(METRICS_BARE)
        sig = self.quiet(srv.read_engine)
        self.assertTrue(sig["reachable"], "metriques absentes != moteur injoignable")
        for key in ("decode_tps", "prefill_tps", "cache_hit_ratio",
                    "queue_wait_seconds_total", "queue_wait_longest_seconds"):
            self.assertIsNone(sig[key], key)
        self.assertNotIn("engine_error", sig)

    def test_cache_hit_ratio_stays_between_0_and_1(self):
        # « plus de couverts que de traites » est le cas NORMAL (le cache couvre la
        # plupart des tokens) : avec le bon denominateur ce n'est pas un depassement.
        set_metrics(METRICS_CACHE_OVERFLOW)
        self.assertAlmostEqual(self.quiet(srv.read_engine)["cache_hit_ratio"],
                               250000 / 350000)
        set_metrics(METRICS_FULL.replace("halogen:prompt_tokens_cached_total 62000",
                                         "halogen:prompt_tokens_cached_total -5"))
        self.assertEqual(self.quiet(srv.read_engine)["cache_hit_ratio"], 0.0)

    def test_zero_prompt_total_does_not_raise(self):
        # aucun token traite sur la periode : tout etait couvert par le cache.
        # Le denominateur n'est plus prompt_total seul, donc plus de None implicite.
        set_metrics(METRICS_ZERO_PROMPT_TOTAL)
        self.assertEqual(self.quiet(srv.read_engine)["cache_hit_ratio"], 1.0)

    def test_two_zero_counters_do_not_raise(self):
        set_metrics(METRICS_ZERO_PROMPT_TOTAL.replace(
            "halogen:prompt_tokens_cached_total 62000",
            "halogen:prompt_tokens_cached_total 0"))
        self.assertIsNone(self.quiet(srv.read_engine)["cache_hit_ratio"])

    def test_labeled_series_are_found(self):
        set_metrics(METRICS_LABELED)
        sig = self.quiet(srv.read_engine)
        self.assertEqual(sig["decode_tps"], 40.5)
        self.assertAlmostEqual(sig["cache_hit_ratio"], 30 / 130)

    def test_engine_down_never_raises(self):
        def boom(url):
            raise urllib.error.URLError("connection refused")
        srv._http_get = boom
        sig = self.quiet(srv.read_engine)
        self.assertFalse(sig["reachable"])
        for key in ("decode_tps", "prefill_tps", "cache_hit_ratio",
                    "queue_wait_seconds_total", "queue_wait_longest_seconds"):
            self.assertIsNone(sig[key], key)
        self.assertIn("engine_error", sig)

    def test_signals_exposed_by_evaluate(self):
        verdict = self.quiet(srv.evaluate)
        self.assertTrue(SIGNAL_KEYS <= set(verdict["signals"]),
                        sorted(SIGNAL_KEYS - set(verdict["signals"])))
        self.assertEqual(verdict["signals"]["decode_tps"], 38.0)


class ServerCase(EngineCase):
    """Cas « bout en bout » : le vrai Handler sur un port libre."""

    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        cls.server.daemon_threads = True
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d" % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def req(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode()
            try:
                return exc.code, json.loads(raw)
            except ValueError:
                return exc.code, {"raw": raw}

    def get(self, path):
        return self.req("GET", path)

    def post(self, path, body=None):
        return self.req("POST", path, body)


class TestPauseHttp(ServerCase):

    def test_pause_shape_when_not_paused(self):
        code, view = self.get("/pause")
        self.assertEqual(code, 200)
        self.assertEqual(set(view), PAUSE_KEYS)
        self.assertEqual(view, {"paused": False, "until": None, "reason": "",
                                "remaining_s": None})

    def test_baseline_is_free(self):
        code, verdict = self.get("/free")
        self.assertEqual(code, 200)
        self.assertTrue(verdict["free"], verdict["reason"])
        self.assertEqual(verdict["reason"], "free")
        self.assertTrue(FREE_KEYS <= set(verdict), sorted(FREE_KEYS - set(verdict)))

    def test_timed_pause_blocks_free(self):
        until = time.time() + 600
        code, view = self.post("/pause", {"until": until, "reason": "maintenance GPU"})
        self.assertEqual(code, 200)
        self.assertEqual(set(view), PAUSE_KEYS)
        self.assertTrue(view["paused"])
        self.assertAlmostEqual(view["until"], until, places=3)
        self.assertEqual(view["reason"], "maintenance GPU")
        self.assertTrue(595 <= view["remaining_s"] <= 600.1, view["remaining_s"])

        code, verdict = self.get("/free")
        self.assertFalse(verdict["free"])
        self.assertEqual(verdict["reason"], "paused")
        self.assertFalse(verdict["idle_ok"])
        self.assertEqual(verdict["pause"]["reason"], "maintenance GPU")

        code, status = self.get("/status")
        self.assertEqual(code, 200)
        self.assertTrue(status["pause"]["paused"])
        self.assertIn("config", status)

    def test_indefinite_pause(self):
        for body in ({}, {"reason": "depannage"}, {"until": None, "reason": "depannage"}):
            code, view = self.post("/pause", body)
            self.assertEqual(code, 200)
            self.assertTrue(view["paused"])
            self.assertIsNone(view["until"])
            self.assertIsNone(view["remaining_s"])
            code, verdict = self.get("/free")
            self.assertEqual(verdict["reason"], "paused")
            self.assertFalse(verdict["free"])

    def test_resume_clears_pause(self):
        self.post("/pause", {"until": time.time() + 600, "reason": "x"})
        code, view = self.post("/resume")
        self.assertEqual(code, 200)
        self.assertEqual(view, {"paused": False, "until": None, "reason": "",
                                "remaining_s": None})
        self.assertEqual(self.get("/pause")[1], view)
        code, verdict = self.get("/free")
        self.assertTrue(verdict["free"], verdict["reason"])
        self.assertEqual(verdict["reason"], "free")
        # /resume sans corps (POST nu) doit rester accepte
        self.assertEqual(self.req("POST", "/resume")[0], 200)

    def test_pause_expires_on_its_own(self):
        self.post("/pause", {"until": time.time() + 1, "reason": "courte"})
        self.assertTrue(self.get("/pause")[1]["paused"])
        time.sleep(1.2)
        view = self.get("/pause")[1]
        self.assertFalse(view["paused"])
        self.assertIsNone(view["remaining_s"])
        verdict = self.get("/free")[1]
        self.assertTrue(verdict["free"], verdict["reason"])
        self.post("/resume")

    def test_bad_until_is_400_and_changes_nothing(self):
        for bad in ("pas-un-nombre", [1, 2], {}):
            code, payload = self.post("/pause", {"until": bad})
            self.assertEqual(code, 400, (bad, payload))
            self.assertIn("error", payload)
            self.assertFalse(self.get("/pause")[1]["paused"])
            self.assertTrue(self.get("/free")[1]["free"])

    def test_idle_clocks_untouched_by_pause(self):
        srv.CONFIG["idle_min_s"] = 0.0
        try:
            now = time.time()
            with srv._LOCK:
                srv.STATE["last_busy_ts"] = now - 100
                srv.STATE["last_activity_ts"] = now - 50
            before = self.get("/free")[1]
            self.assertTrue(before["free"])
            self.assertTrue(before["idle_ok"])
            self.assertGreaterEqual(before["idle_for_s"], 100)
            self.assertGreaterEqual(before["calm_for_s"], 50)

            self.post("/pause", {"until": time.time() + 600, "reason": "horloges"})
            after = self.get("/free")[1]
            self.assertEqual(after["reason"], "paused")
            self.assertFalse(after["idle_ok"])
            # les deux horloges continuent de monter : la pause ne les recale pas
            self.assertGreaterEqual(after["idle_for_s"], before["idle_for_s"])
            self.assertGreaterEqual(after["calm_for_s"], before["calm_for_s"])
            self.assertLess(abs(after["idle_for_s"] - before["idle_for_s"]), 5)
            self.assertLess(abs(after["calm_for_s"] - before["calm_for_s"]), 5)
        finally:
            self.post("/resume")
            srv.CONFIG["idle_min_s"] = 30.0

    def test_pause_persisted_and_restored(self):
        until = time.time() + 3600
        self.post("/pause", {"until": until, "reason": "restart"})
        payload = json.loads(Path(srv.CONFIG["state_file"]).read_text(encoding="utf-8"))
        self.assertIsInstance(payload["pause"], dict)
        self.assertAlmostEqual(payload["pause"]["until"], until, places=2)
        self.assertEqual(payload["pause"]["reason"], "restart")
        self.assertIn("lease", payload)  # la cle du bail n'a pas bouge

        # « restart » : memoire vide, puis _restore() comme au demarrage du service
        with srv._LOCK:
            srv.STATE["pause"] = None
        self.assertFalse(self.get("/pause")[1]["paused"])
        with redirect_stdout(io.StringIO()):
            srv._restore()
        view = self.get("/pause")[1]
        self.assertTrue(view["paused"])
        self.assertEqual(view["reason"], "restart")
        self.assertEqual(self.get("/free")[1]["reason"], "paused")
        self.post("/resume")
        self.assertFalse(self.get("/pause")[1]["paused"])
        after = json.loads(Path(srv.CONFIG["state_file"]).read_text(encoding="utf-8"))
        self.assertIsNone(after["pause"])

    def test_pause_wins_over_lease_and_leaves_it_intact(self):
        code, lease = self.post("/lease", {"agent": "test-agent", "task": "t", "ttl": 600})
        self.assertEqual(code, 200, lease)
        try:
            self.assertEqual(self.get("/free")[1]["reason"],
                             "lease_held:test-agent")
            self.post("/pause", {"until": time.time() + 600, "reason": "priorite"})
            verdict = self.get("/free")[1]
            self.assertEqual(verdict["reason"], "paused")
            self.assertIsNotNone(verdict["lease"], "la pause ne doit pas liberer le bail")
            self.assertEqual(self.get("/lease")[1]["lease"]["agent"], "test-agent")
            self.post("/resume")
            self.assertEqual(self.get("/free")[1]["reason"], "lease_held:test-agent")
        finally:
            self.req("DELETE", "/lease/%s" % lease["id"])

    def test_pause_events_are_logged(self):
        self.post("/pause", {"until": time.time() + 60, "reason": "journal"})
        self.post("/resume")
        events = [e["event"] for e in self.get("/history")[1]["events"]]
        self.assertIn("pause_set", events)
        self.assertIn("pause_cleared", events)


if __name__ == "__main__":
    unittest.main(verbosity=2)
