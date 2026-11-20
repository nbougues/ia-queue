# ia-queue — GPU admission arbiter + idle-slot agents for Hermes

Two things live here, and they run on different machines:

1. **`ia-free`** — a small, stdlib-only HTTP arbiter that answers one question with
   evidence: *"may this session use the inference engine right now?"* It runs on the
   **inference host**, next to the engine it measures.
2. **The idle-slot workers** — Hermes agents that wake up on a timer, ask the
   arbiter, and only then spend tokens on a queued non-urgent task. They run on
   each **agent host**.

The reason both exist: several independent Hermes instances share one inference
engine and never talk to each other. Anything that coordinates them has to live
somewhere they all look at. That place is the arbiter.

```
        agent host                                inference host
  ┌─────────────────────────────────────┐      ┌──────────────────────────────────┐
  │ cron Hermes (every 10m, --no-agent) │      │  inference engine :8731          │
  │   └─ ia_slot.py ─► ia_slot_gate.py  │      │    /metrics  /slots  /cache      │
  │        queue non-empty? spawn ──────┼──┐   │                                  │
  │        (never sleeps in the tick)   │  │   │  ia-free :8765  ◄── reads engine │
  │   ia_slot_worker.py (detached)      │  └──►│    GET /free  POST /lease        │
  │     jitter → /free → /lease → task  │      │    POST /warmth  GET /status     │
  │     → real Hermes session → release │      │                                  │
  └─────────────────────────────────────┘      └──────────────────────────────────┘
              ▲                                              ▲
              │  llm_execution middleware asks /free         │  same /free endpoint
        hermes-gpu-gate plugin ─────────────────────────────┘
        (per-session priority tags: low / ultralow)
```

## Packaging — what is where, and what runs where

| File | Role | Runs on |
|---|---|---|
| `ia_free_server.py` | the arbiter: `/free`, `/warmth`, `/lease`, `/pause`. Stdlib only, no deps | **inference host**, as `ia-free.service` |
| `ia-free.service` | systemd unit (`DynamicUser`, state in `/var/lib/ia-free`, `MemoryMax=96M`) | inference host |
| `deploy-inference-host.sh` | installs/updates the service on the inference host and restarts it | run from an agent host |
| `ia_slot_common.py` | shared config, HTTP, locks, task-queue client, `run_hermes()` | agent hosts |
| `ia_slot_gate.py` | the cron tick: decides whether a slot is worth taking, spawns the worker | agent hosts |
| `ia_slot_worker.py` | the work: jitter → `/free` → lease → claim task → run session → release | agent hosts (detached) |
| `hermes-scripts/ia_slot.py` | wrapper installed into `<HERMES_HOME>/scripts/` | each profile |
| `install-agents.sh` | installs the wrapper + the cron job per Hermes profile | run from an agent host |
| `config.example.json` | shared knobs + one `agents.<profile>` entry each. **Copy to `config.json` and edit** | agent hosts |
| `tests/` | unit tests; the engine is simulated, no network and no GPU needed | anywhere |
| `SETUP.md`, `PROMPT-INSTALL-HERMES.md` | install procedure; prompts to hand another agent | — |

Runtime data is deliberately **not** in the repo: `config.json` (your real
queues and paths), `state/`, `logs/`, delivered artefacts.

## The admission contract (the part worth reading)

`GET /free` returns `free: true|false` plus the evidence. Two **independent**
conditions must hold:

1. **Queue depth** — `inflight <= IA_FREE_MAX_INFLIGHT` (default 3). This is a
   queue guard, *not* a capacity limit.
2. **KV headroom** — `kv_free_positions >= IA_FREE_KV_RESERVE_POSITIONS`
   (default 157 000), i.e. there is room in the engine's KV pool for one more
   full context without evicting a resident one.

Condition 2 is the one that matters, and it is what condition 1 used to
approximate badly. The failure mode is worth being precise about, because it is
counter-intuitive: when the sum of resident contexts exceeds the KV pool, the
engine keeps *nothing*. Every turn then pays a full re-prefill of its whole
context. On a deployment with a pool holding roughly five 150 k-token sessions,
that measured as **0 % prefix-cache retention and ~90 s per turn to emit 96
tokens**; after sizing the pool for the actual number of concurrent sessions,
**100 % retention and well under a second**. An eviction is not "a bit slower",
it is a complete GPU re-read of the context — so the admission question is never
"how many requests are running" but "is there room for this context".

The arbiter cannot know the size of a request that has not arrived yet, so it
reserves one typical context and refuses below that. **It fails open** when the
engine publishes no counters: an arbiter that cannot measure has no business
blocking work.

Two clocks, easy to confuse, both exposed:

| Field | Measures | Reset by |
|---|---|---|
| `calm_for_s` | since the last **contention** | `inflight > max`, engine unreachable, lease release |
| `idle_for_s` | since the last **trace of engine activity** | any request in flight, `busy_for_seconds > 0`, or `requests_total` incrementing |

A single lonely request does **not** move `calm_for_s` but does zero `idle_for_s`.
"Quiet for N seconds" must be read from `idle_for_s` (`idle_ok` =
`idle_for_s >= IA_FREE_IDLE_MIN_S`), never from `calm_for_s`. An internal sampler
polls the engine every `IA_FREE_IDLE_SAMPLE_S` so the clock advances with no clients.

### Cache warmth

`POST /warmth` lets each Hermes instance report what its last call actually hit
(`prompt_tokens`, `cached_tokens` — numbers only the caller can see). The arbiter
answers `warmth` on `/free?session=…`: a session is `hot` only if its ratio is
good, it was seen recently, **and** the engine's global `cache_evicted_total` has
not moved since. That third term is what makes it correct across instances that
cannot see each other — nothing tells session A that instance B just prefilled
40 k tokens, but both read the same counter. All counters are monotonic only per
engine generation, so the arbiter tracks an `engine_epoch` and wipes the table
when the engine restarts; otherwise a session that was warm just before a restart
stays warm forever, which is exactly the false positive that sends a cold session
charging ahead.

## Articulation with `hermes-gpu-gate`

Clean split of responsibility, and the boundary is the `/free` endpoint:

| | **ia-free** (this repo) | **hermes-gpu-gate** |
|---|---|---|
| runs | on the inference host, one process | inside each Hermes process (plugin) |
| knows | engine metrics, leases, every instance's warmth | the session: its tag, platform, cwd, wait budget |
| decides | *is there room* | *should this session wait, and how long* |
| acts | answers HTTP, grants leases | waits in `llm_execution` middleware, then proceeds anyway at budget exhaustion |

The gate never vetoes — it waits, and at the end of its budget it proceeds
regardless, because a turn that waits forever gets killed by a watchdog and
produces nothing. The arbiter never blocks: it only answers.

How a slot worker's own session gets throttled: the gate resolves a mode per
session (manual tag → `mode_by_cwd` → `mode_by_platform` → unattended →
`default_mode`). The worker launches sessions **from its `workdir`**, which is
what the gate can actually see — the middleware runs in the gateway process and
never sees the launcher's environment. So the rule is by working directory:

```yaml
plugins.entries.gpu-gate.settings.mode_by_cwd:
  /path/to/ia-queue: ultralow
```

`ultralow` requires the engine quiet for `idle_min_s` at turn entry, then only
`free` afterwards. Interactive profiles are better off at `default_mode: normal`
and cron at `low`: with enough KV headroom there is nothing left to protect by
serialising interactive sessions. Idle-slot work stays `ultralow` — those are the
big cold prefills, the only real GPU contention left. `gpu-gate/apply-settings.py`
migrates profiles.

The **lease** is the other half: `/free` is an observation, so two callers can
both see "free" and both go. Real serialisation of background work is the single
global lease (`POST /lease`, TTL 45 min, renewed every 5 min, auto-expires if the
worker dies). One background task at a time, across every Hermes instance.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/free` | the verdict (`?session=&tokens=` adds that session's warmth) |
| `GET` | `/status` | verdict + full signal dump |
| `GET` | `/warmth` | the whole warmth table |
| `POST` | `/warmth` | report one call's `{session, prompt_tokens, cached_tokens}` |
| `POST` | `/lease`, `/lease/renew` | take / renew the single global lease |
| `DELETE` | `/lease/<id>` | release |
| `POST` | `/pause`, `/resume` | hold everything, with a reason |
| `GET` | `/history`, `/healthz` | recent decisions, liveness |

Unauthenticated — **private network only**. It reads the engine's `/metrics` and
`/slots` plus `/sys/class/drm/card*/device/gpu_busy_percent`.

## Configuration

Service knobs (systemd `Environment=`, read once at import):

| Knob | Default | Effect |
|---|---|---|
| `IA_FREE_MAX_INFLIGHT` | 3 | queue guard, not capacity |
| `IA_FREE_KV_GATE` | 1 | enable the KV headroom rule (0 = purely informational) |
| `IA_FREE_KV_RESERVE_POSITIONS` | 157000 | room required for one more context |
| `IA_FREE_IDLE_MIN_S` | 30 | quiet required for `idle_ok` |
| `IA_FREE_IDLE_SAMPLE_S` | 2 | engine sampling period (0 = only on client calls) |
| `IA_FREE_COOLDOWN_S` | 0 | off: the lease serialises background work, not this |
| `IA_FREE_GPU_MAX` | -1 | GPU % is a signal, not a refusal |
| `IA_FREE_LEASE_TTL` | 2700 | lease TTL |

Worker knobs live in `config.json` (copy `config.example.json`): `server`,
`jitter_max_s`, `lease_ttl_s`, `max_task_minutes`, `window`, and one
`agents.<profile>` entry per Hermes profile, each with its own task queue and
write scope.

## Install

```bash
# arbiter, on the inference host (sudo)
./deploy-inference-host.sh you@inference-host

# workers, on each agent host: wrapper + cron job per Hermes profile
./install-agents.sh <profile> [<profile> …]
```

`SETUP.md` walks through both, including wiring the task queue per profile.

## Tests

```bash
python3 -m unittest discover -s tests        # no network, no GPU
python3 ia_free_server.py --once             # evaluate locally, print, exit
```

The engine is simulated by a fake `_http_get` returning real captured Prometheus
text, so the metric formulas are pinned against numbers read off a live engine.
Note that `CONFIG` is frozen at import: pin knobs by assigning `CONFIG` in
`setUp`, not via `os.environ`, or a later test module silently changes nothing.

## Design choices worth knowing

- **A cron tick that does nothing costs zero tokens** (`--no-agent`) and never
  sleeps; the detached worker does the sleeping, so the next tick stays on time.
- **The gate checks the queue before detaching** — no worker woken for an empty queue.
- **One global lease, not one per profile**: the engine is shared, so background
  work is serialised across every Hermes instance, while each profile keeps its
  own task queue and its own write scope.
- **Everything fails open.** Unreachable engine, missing counters, dead arbiter:
  work proceeds. A broken admission layer must degrade to "no admission layer",
  never to "nothing runs".
