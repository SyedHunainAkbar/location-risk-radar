# Design: LLM Gateway

## Overview

`src/lrr/gateway/` is a small package that centralizes every LLM call. Callers ask
for a role (for example `rag_answer`); the gateway resolves the role to an ordered
fallback chain from `config/gateway.yaml`, verifies/repairs model ids, checks health,
calls the first healthy provider, retries transient errors, validates JSON when a
schema is given, caches results, and records a ledger row. `src/lrr/llm.py` becomes a
thin compatibility shim re-exporting the facade so existing modules keep working.

Design constraints honored: keys only from env or `st.secrets`, never logged or
cached; `mock` is always the terminal fallback so a keyless run answers; yaml and
pydantic are used when present with safe fallbacks so tests never hard-require them.

## Package layout

```
src/lrr/gateway/
  __init__.py        # public facade: chat, achat, Gateway, get_gateway, GatewayResult
  providers.py       # provider registry, OpenAI-compatible client factory, mock
  routing.py         # load gateway.yaml, Role -> chain of RouteEntry
  redaction.py       # logging filter masking sk-/sk_/nvapi-
  health.py          # per-provider health check, 10-min cache, voyager auto-disable
  ratelimit.py       # per-provider token-bucket limiter
  cache.py           # disk cache keyed by hash(provider, model, messages, params)
  ledger.py          # usage ledger (parquet offline, list in-memory for the app)
  verify.py          # /models verification + artifacts/gateway_models.json
  core.py            # the Gateway: routing + retry + JSON repair + None handling
```

## providers.py

```python
PROVIDERS = {"openai", "nvidia", "voyager", "mock"}
PROVIDER_BASE_URLS, PROVIDER_KEY_ENV
def get_secret(name) -> str | None          # env, then st.secrets; never cached
def make_client(provider)                   # OpenAI(base_url, api_key); lazy import
def mock_response(messages, model) -> str   # deterministic (sha256 of input)
def estimate_tokens(text) -> int            # cheap heuristic when usage is absent
```

## routing.py

```python
@dataclass RouteEntry: provider, model, temperature, max_tokens, timeout
def load_routes(path=config.GATEWAY_YAML) -> dict[str, list[RouteEntry]]
DEFAULT_ROUTES                               # same roles as yaml, used if file missing
```

The yaml lists each role as an ordered array of `provider:model` strings plus shared
`temperature`, `max_tokens`, `timeout`. `load_routes` parses it (pyyaml) and falls
back to `DEFAULT_ROUTES` if the file or pyyaml is unavailable.

## redaction.py

A `logging.Filter` that regex-replaces `sk-[A-Za-z0-9]+`, `sk_[A-Za-z0-9]+`, and
`nvapi-[A-Za-z0-9]+` with a masked token in `record.msg` and `record.args`. Installed
on the `lrr.gateway` logger at import. A `redact(text)` helper is reused by tests.

## health.py

```python
def check_provider(provider, now=...) -> bool      # cached 10 min
def healthy_providers() -> set[str]                # voyager removed if unreachable
```

Health is a cheap client construction + a guarded `/models` ping. Results cache for
`HEALTH_TTL_SECONDS` (600). `voyager` is auto-disabled (treated unhealthy) when the
ping fails, since it needs the ASU VPN.

## cache.py

```python
def cache_key(provider, model, messages, params) -> str   # sha256 hex
def get(key) -> dict | None
def put(key, value) -> None
```

JSON files under `config.LLM_CACHE_DIR` (`.cache/llm`, gitignored). Keys never
include secrets. A `params` dict carries temperature/max_tokens/json-schema name only.

## ledger.py

```python
@dataclass LedgerRow: role, provider, model, latency_s, prompt_tokens,
                      completion_tokens, est_cost_usd, cache_hit, fallback_used
class Ledger: append(row); to_frame(); flush(path)
```

Offline, `flush` appends to `artifacts/llm_ledger.parquet`. In the app, the Gateway
keeps an in-memory `Ledger` the sidebar reads from session state. Cost uses a small
per-model price table with a 0.0 default for unknown/mock.

## verify.py

```python
def verify_models(routes, write=True) -> dict[str, list[str]]
```

For each live provider, call `/models`, collect available ids, and for each configured
model id substitute an available id when the configured one is absent (prefer a close
match, else the first available). Writes `artifacts/gateway_models.json`:
`{provider: [verified ids...], "substitutions": {...}, "checked_at": iso}`. Never
raises on an unreachable provider; it just records what it could confirm.

## core.py (the Gateway)

```python
@dataclass GatewayResult: text, provider, model, role, cache_hit, fallback_used,
                          latency_s, parsed (optional pydantic model)
class Gateway:
    def chat(messages, role, schema=None, **overrides) -> GatewayResult
    async def achat(...) -> GatewayResult
```

Algorithm for `chat`:
1. Resolve the role to its fallback chain; drop entries whose provider is unhealthy
   (except `mock`, always kept).
2. Compute the cache key; on hit, record a ledger row with `cache_hit=True` and
   return.
3. Walk the chain. For each entry: apply the per-provider rate limiter and timeout,
   call the client, retry on 429/5xx with exponential backoff + jitter, and on empty
   or `None` content retry with a larger `max_tokens`. On success, break; mark
   `fallback_used` if the entry was not the first.
4. If a `schema` (pydantic model) is given, validate; on failure send one repair
   message ("return valid JSON matching this schema") and revalidate once.
5. Record the ledger row (role, provider, model, latency, tokens, est cost, cache
   hit, fallback) and cache the raw text.

`mock` short-circuits network concerns and returns a deterministic string (and valid
JSON when a schema is requested, built from the schema's fields).

## Facade and back-compat

`gateway/__init__.py` exposes `chat`, `achat`, `get_gateway`, `GatewayResult`. The old
`lrr.llm` module is rewritten to re-export these and keep
`chat(messages, model=None, temperature=..., max_tokens=..., provider=None)` working
by mapping `provider`/`model` to a one-entry ad hoc chain (and the default role when
none is given). This keeps `topics`, `aspects`, and `rag` unchanged while they can opt
into roles later.

## App integration

A helper `gateway.app_panel(st)` renders the sidebar: provider health dots, a
Live/Cached radio (Cached forces cache-only + mock), and it exposes the latest
`GatewayResult` so each response can show a badge "answered by {provider}:{model}".
The app stores the `Ledger` in `st.session_state`.

## Testing strategy

All offline, injected fakes, no network:
- Fallback order: a fake client that fails for the first k providers; assert the
  answering provider is the first healthy one and `fallback_used` is set.
- Cache hit: two identical calls; the second has `cache_hit=True` and does not call
  the client.
- Schema repair: a fake that returns bad JSON then valid JSON; assert one repair.
- Redaction: `redact("key sk-abc123...")` masks the token; a log record is scrubbed.
- Voyager disabled: a health probe that fails for voyager removes it from the chain.
