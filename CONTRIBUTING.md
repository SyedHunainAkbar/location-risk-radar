# Contributing

Thanks for your interest. This repository accompanies a graduate course project, so
contributions are mostly from the team, but the conventions below keep the history
clean.

## Branches

- `main` is protected and always green. Do not commit to it directly.
- Branch from `main` using `type/short-description`, for example
  `feat/portfolio-map` or `fix/cohort-reconciliation`.

## Commits

We use Conventional Commits:

- `feat:` a new feature
- `fix:` a bug fix
- `docs:` documentation only
- `test:` tests only
- `ci:` CI or tooling
- `refactor:`, `perf:`, `chore:` as appropriate

Keep each commit scoped to one concern with a clear message.

## Before you push

```bash
pip install -r requirements.txt
pip install pre-commit ruff pytest pytest-cov
pre-commit install
make lint
make test
```

- `ruff` and `ruff-format` must pass.
- `pytest` must pass; coverage for `src/lrr` must stay at or above 80 percent.
- The secret scanners (gitleaks, detect-secrets) must find nothing. Never commit a
  real API key; keys come only from the environment or `st.secrets`.

## Pull requests

Open a PR against `main`, fill in the template, and wait for CI to pass. A green CI
run is required before merge.
