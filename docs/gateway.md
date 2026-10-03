# LLM Gateway

Every model call in the project routes through `src/lrr/gateway`. One entry point,
`chat(messages, role=..., schema=...)`, resolves a task role to an ordered provider
fallback chain, checks provider health, retries transient failures, validates JSON,
caches responses, and records a usage ledger.

## Providers

| Provider | Base URL | Key env | Notes |
|---|---|---|---|
| `openai` | SDK default | `OPENAI_API_KEY` | |
| `nvidia` | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` | default in the app |
| `voyager` | `https://openai.rc.asu.edu/v1` | `VOYAGER_API_KEY` | ASU VPN, local only |
| `mock` | none | none | deterministic, offline, terminal fallback |

Keys are read only from environment variables or `st.secrets`. They are never
logged, printed, or cached. A logging filter masks `sk-`, `sk_`, and `nvapi-`
patterns.

## Routing

`config/gateway.yaml` maps roles to fallback chains, each a `provider:model` with
temperature, max tokens, and timeout. Roles: `quant_analyst`, `voice_of_customer`,
`risk_auditor`, `topic_labeler`, `rag_answer`, `rag_judge`. The gateway tries entries
in order and uses the first that succeeds; `mock` is always the terminal fallback so
a keyless run still answers.

## Reliability

- Per-provider health check cached for 10 minutes; `voyager` auto-disabled when
  unreachable.
- Retries on HTTP 429 and 5xx with exponential backoff and jitter.
- Per-provider rate limiter and per-call timeout.
- JSON mode with pydantic schema validation and one repair retry.
- Empty content from reasoning models is retried with a larger token budget.
- Disk cache keyed by `hash(provider, model, messages, params)` under `.cache/llm`.
- Async API (`achat`) for parallel calls.
- A usage ledger records role, provider, model, latency, tokens, estimated cost,
  cache-hit, and whether a fallback was used.

Model ids are verified against each provider's `/models` endpoint, with substitution
for unavailable ids; the verified list is written to `artifacts/gateway_models.json`.
