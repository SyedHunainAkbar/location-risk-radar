# Requirements: LLM Gateway

## Introduction

We replace the single `src/lrr/llm.py` with a `src/lrr/gateway/` package that every
LLM caller uses (topic labeling, aspect agents, RAG answer and judge, LA5 helpers).
The gateway routes task roles to ordered provider fallback chains, verifies model ids
against each provider, checks provider health, retries transient failures, enforces
JSON schemas with a repair retry, caches to disk, exposes a sync and async API, and
records a usage ledger. Keys are read only from the environment or `st.secrets` and
are never logged, printed, or cached.

## Requirements

### Requirement 1: Providers

#### Acceptance Criteria
1. The gateway SHALL support four providers via OpenAI-compatible clients:
   - `openai`: default base URL, key `OPENAI_API_KEY`.
   - `nvidia`: `https://integrate.api.nvidia.com/v1`, key `NVIDIA_API_KEY`.
   - `voyager`: `https://openai.rc.asu.edu/v1`, key `VOYAGER_API_KEY` (ASU VPN, local
     only).
   - `mock`: deterministic offline responses for tests and a keyless demo.
2. Keys SHALL be read only from environment variables or `st.secrets`.
3. Keys SHALL never be logged, printed, or written to the cache or ledger.

### Requirement 2: Key redaction

#### Acceptance Criteria
1. The gateway logger SHALL install a redaction filter that masks `sk-`, `sk_`, and
   `nvapi-` patterns in every log record before it is emitted.
2. The redaction SHALL apply to messages and formatted args.

### Requirement 3: Role-based routing with fallback chains

#### Acceptance Criteria
1. `config/gateway.yaml` SHALL map task roles to ordered fallback chains, each entry a
   `provider:model` with `temperature`, `max_tokens`, and `timeout`.
2. Roles SHALL include at least: `quant_analyst`, `voice_of_customer`,
   `risk_auditor`, `topic_labeler`, `rag_answer`, `rag_judge`.
3. A request for a role SHALL try chain entries in order and use the first that
   succeeds; `mock` SHALL be the terminal fallback so a keyless run always answers.
4. The entry that answered SHALL be reported to the caller (provider and model).

### Requirement 4: Model-id verification

#### Acceptance Criteria
1. Before finalizing, the gateway SHALL verify each configured model id by calling the
   provider's `/models` endpoint and SHALL fall back to an available id when the
   configured id is absent.
2. The verified per-provider model list SHALL be written to
   `artifacts/gateway_models.json`.
3. Verification SHALL be skippable offline (mock), and failure to reach a provider
   SHALL NOT crash the gateway.

### Requirement 5: Reliability features

#### Acceptance Criteria
1. The gateway SHALL run a per-provider health check at startup, cached for 10
   minutes; `voyager` SHALL be auto-disabled when unreachable.
2. The gateway SHALL retry on HTTP 429 and 5xx with exponential backoff and jitter.
3. The gateway SHALL enforce a per-provider rate limiter and a per-call timeout.
4. The gateway SHALL support JSON mode with pydantic schema validation and exactly one
   repair retry on invalid JSON.
5. The gateway SHALL handle `None` or empty content from reasoning models by retrying
   with a larger token budget.

### Requirement 6: Caching and async

#### Acceptance Criteria
1. The gateway SHALL cache responses on disk keyed by
   `hash(provider, model, messages, params)` under `.cache/llm` (gitignored).
2. The gateway SHALL expose an async API (`achat`) for parallel calls.

### Requirement 7: Usage ledger

#### Acceptance Criteria
1. The gateway SHALL record a ledger row per call with role, provider, model, latency,
   prompt tokens, completion tokens, estimated cost, cache-hit flag, and whether a
   fallback was used.
2. Offline runs SHALL append to `artifacts/llm_ledger.parquet`; the app SHALL keep the
   ledger in session state.

### Requirement 8: App integration

#### Acceptance Criteria
1. The app SHALL show an "LLM Gateway" sidebar panel with per-provider health, a
   Live/Cached mode switch, and per-response badges naming the answering provider and
   model.

### Requirement 9: Backward compatibility and migration

#### Acceptance Criteria
1. Existing callers (`topics`, `aspects`, `rag`) SHALL continue to work; the old
   `lrr.llm.chat(messages, model, temperature, max_tokens, provider)` signature SHALL
   remain callable as a thin shim over the gateway.
2. `src/lrr/llm.py` SHALL be replaced by `src/lrr/gateway/` with `lrr.llm` kept as a
   compatibility module re-exporting the facade.

### Requirement 10: Tests

#### Acceptance Criteria
1. Tests SHALL cover: fallback order on simulated failures, a cache hit, JSON schema
   repair, key redaction, and `voyager` disabled when unreachable.
2. Tests SHALL run with no network and no API keys using the `mock` provider and
   injected fakes.
