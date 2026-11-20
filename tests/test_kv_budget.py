#!/usr/bin/env python3
"""Tests du budget en positions KV (ia-free-server).

    python3 -m unittest discover -s tests          # ou : python3 tests/test_kv_budget.py

Ce que ces tests verrouillent, et pourquoi :

  * La contrainte mesurée sur le vrai moteur : 5 sessions de 148k pour un pool de
    786k donnaient 0 % de rétention et ~88 s de re-prefill par tour. La question
    pertinente n'est donc pas « combien de requêtes tournent » (le moteur a 4 slots
    et les sert bien de front) mais « reste-t-il la place d'un contexte de plus ».
  * `kv_tight` doit refuser d'admettre quand il ne reste pas la réserve, MÊME si le
    moteur est parfaitement calme : un moteur au repos mais plein est exactement le
    cas qui faisait effondrer le débit.
  * Fail-open : sans compteurs, on admet. Un arbitre aveugle n'a pas le droit
    d'immobiliser du travail.
  * Le garde-fou `inflight` reste premier : une raison plus grave passe avant.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

# a faire AVANT l'import : CONFIG est fige a l'import
_TMP = Path(tempfile.mkdtemp(prefix="ia-free-kv-"))
os.environ["IA_FREE_STATE_FILE"] = str(_TMP / "state.json")
os.environ["IA_FREE_IDLE_SAMPLE_S"] = "0"
os.environ["IA_FREE_MAX_INFLIGHT"] = "3"
os.environ["IA_FREE_KV_GATE"] = "1"
os.environ["IA_FREE_KV_RESERVE_POSITIONS"] = "157000"

import ia_free_server as srv  # noqa: E402

_LOG_SINK = (_TMP / "service.log").open("w", encoding="utf-8")
_LOG_LOCK = threading.Lock()
_real_log = srv.log


def _silent_log(event, **fields):
    with _LOG_LOCK, redirect_stdout(_LOG_SINK):
        _real_log(event, **fields)


srv.log = _silent_log

POOL = 1_572_864


def metrics(used: int, pool: int = POOL, processing: int = 0) -> str:
    return ("llamacpp:requests_processing %d\n"
            "llamacpp:requests_deferred 0\n"
            "llamacpp:kv_cache_tokens %d\n"
            "halogen:kv_pool_positions %d\n"
            "halogen:requests_total 100\n"
            "halogen:busy_for_seconds 0\n" % (processing, used, pool))


def install(metrics_text):
    def fake_http_get(url: str) -> str:
        if url.endswith("/slots"):
            return "[]"
        if url.endswith("/metrics"):
            return metrics_text
        raise urllib.error.URLError("test: no such url %s" % url)
    srv._http_get = fake_http_get


class KvBudgetCase(unittest.TestCase):
    def setUp(self):
        self._saved = {k: srv.CONFIG[k] for k in
                       ("kv_gate", "kv_reserve_positions", "max_inflight")}
        srv.CONFIG["kv_gate"] = True
        srv.CONFIG["kv_reserve_positions"] = 157_000
        srv.CONFIG["max_inflight"] = 3

    def tearDown(self):
        srv.CONFIG.update(self._saved)

    def verdict(self):
        return srv.evaluate(session="test")

    # ------------------------------------------------------------ calcul de marge
    def test_headroom_is_pool_minus_used(self):
        install(metrics(used=1_024_768))
        v = self.verdict()
        self.assertEqual(v["signals"]["kv_free_positions"], POOL - 1_024_768)
        self.assertEqual(v["signals"]["kv_reserve_positions"], 157_000)

    def test_admits_while_a_context_fits(self):
        # 5 sessions de 148k residentes : il reste ~831k, largement de quoi en loger une
        install(metrics(used=5 * 148_000))
        v = self.verdict()
        self.assertTrue(v["signals"]["kv_ok"])
        self.assertTrue(v["free"], v["reason"])

    def test_refuses_when_no_room_for_one_more_context(self):
        # 10 sessions de 157k : le pool est plein, le moteur est calme, et c'est
        # précisément le cas qu'il faut refuser (sinon eviction => ~88 s de re-prefill)
        install(metrics(used=POOL - 100_000, processing=0))
        v = self.verdict()
        self.assertFalse(v["free"])
        self.assertEqual(v["reason"], "kv_tight")
        self.assertFalse(v["signals"]["kv_ok"])

    def test_refusal_is_independent_of_engine_being_idle(self):
        install(metrics(used=POOL - 1_000, processing=0))
        v = self.verdict()
        self.assertEqual(v["reason"], "kv_tight")
        self.assertEqual(v["signals"]["inflight"], 0)

    def test_reserve_is_configurable(self):
        srv.CONFIG["kv_reserve_positions"] = 10_000
        install(metrics(used=POOL - 50_000))
        self.assertTrue(self.verdict()["free"])
        srv.CONFIG["kv_reserve_positions"] = 200_000
        self.assertFalse(self.verdict()["free"])

    def test_gate_can_be_switched_off(self):
        srv.CONFIG["kv_gate"] = False
        install(metrics(used=POOL - 1))
        v = self.verdict()
        self.assertTrue(v["free"], "kv_gate=0 doit rendre le budget purement informatif")
        # le signal reste publie : on peut observer sans appliquer
        self.assertFalse(v["signals"]["kv_ok"])

    # ------------------------------------------------------------------- fail-open
    def test_fail_open_when_engine_publishes_nothing(self):
        install("llamacpp:requests_processing 0\nhalogen:requests_total 1\n")
        v = self.verdict()
        self.assertTrue(v["free"], "sans compteurs, l'arbitre n'a pas le droit de bloquer")
        self.assertIsNone(v["signals"]["kv_free_positions"])
        self.assertTrue(v["signals"]["kv_ok"])

    def test_fail_open_when_only_pool_size_known(self):
        install("llamacpp:requests_processing 0\nhalogen:kv_pool_positions %d\n" % POOL)
        self.assertTrue(self.verdict()["free"])

    # ------------------------------------------------------------- priorite reasons
    def test_inflight_reason_wins_over_kv_tight(self):
        install(metrics(used=POOL - 1, processing=9))
        v = self.verdict()
        self.assertEqual(v["reason"], "inflight_above_limit")

    def test_kv_fields_are_exposed_in_signals(self):
        install(metrics(used=1_000))
        v = self.verdict()
        for key in ("kv_free_positions", "kv_reserve_positions", "kv_ok"):
            self.assertIn(key, v["signals"], key)


class DefaultConfigCase(unittest.TestCase):
    def test_source_defaults_match_the_engine_reality(self):
        # Le defaut historique etait max_inflight=1 : l'arbitre etait 4x plus severe
        # que le moteur (4 slots, jusqu'a 8 flux en rounds conjoints), et kv_gate
        # n'existait pas. Verrouille ici pour qu'on ne « repare » pas en arriere,
        # independamment de l'environnement de ce fichier de test.
        src = (ROOT / "ia_free_server.py").read_text(encoding="utf-8")
        self.assertIn('_env("IA_FREE_MAX_INFLIGHT", "3")', src)
        self.assertIn('_env("IA_FREE_KV_GATE", "1", bool)', src)
        self.assertIn('_env("IA_FREE_KV_RESERVE_POSITIONS", "157000")', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
