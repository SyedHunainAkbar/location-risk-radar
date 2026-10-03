# Tasks: LLM Gateway

- [ ] 1. Config: `config/gateway.yaml` + gateway constants in `config.py`
  - Role fallback chains; cache dir, health TTL, ledger path, models-json path.
  - _Requirements: 3.1, 3.2, 5.1, 6.1, 7.2_

- [ ] 2. `gateway/redaction.py`, `gateway/providers.py`
  - Redaction filter; provider registry, client factory, mock, secret reader.
  - _Requirements: 1.1-1.3, 2.1, 2.2_

- [ ] 3. `gateway/routing.py`, `gateway/health.py`, `gateway/ratelimit.py`
  - Load routes with default fallback; health check cached 10 min, voyager
    auto-disable; per-provider token bucket.
  - _Requirements: 3.1, 3.3, 5.1, 5.3_

- [ ] 4. `gateway/cache.py`, `gateway/ledger.py`
  - Disk cache keyed by hash; ledger rows + parquet flush / in-memory.
  - _Requirements: 6.1, 7.1, 7.2_

- [ ] 5. `gateway/core.py` + `gateway/__init__.py` facade
  - Routing + retry/jitter + None handling + JSON schema + one repair; chat/achat.
  - _Requirements: 3.3, 3.4, 5.2, 5.4, 5.5, 6.2_

- [ ] 6. `gateway/verify.py` + `artifacts/gateway_models.json`
  - /models verification with substitution; write verified list.
  - _Requirements: 4.1, 4.2, 4.3_

- [ ] 7. Back-compat: rewrite `lrr/llm.py` as a shim; keep topics/aspects/rag working
  - _Requirements: 9.1, 9.2_

- [ ] 8. App panel helper + tests + pytest
  - Sidebar health/mode/badges helper; tests for fallback, cache, repair,
    redaction, voyager disabled.
  - _Requirements: 8.1, 10.1, 10.2_
