# Secrets Audit

**Date:** 2026-10-03
**Scope:** working tree and full git history for `sk-`, `sk_`, `nvapi-` patterns and
any committed `.env`.

## Result: CLEAN (no real secrets, no git history)

### Git history
The working tree is **not a git repository** (`git rev-parse` returns
`fatal: not a git repository`; no `.git` directory). There is no commit history and
no blobs, so the history scan is vacuously clean. `git filter-repo` is **not
applicable**, there is nothing to rewrite and nothing to force-push.

### Working tree
A full-tree scan for `sk-[A-Za-z0-9]{16}`, `nvapi-[A-Za-z0-9]{16}`, `sk_live`, and
`sk_test` produced only three benign matches:

| Path | Match | Verdict |
|------|-------|---------|
| `.venv/.../pip-26.2.1.dist-info/RECORD` | `sha256=...` containing a coincidental `sk-` substring | Not a secret; `.venv` is gitignored |
| `tests/test_gateway.py` | `sk_live_abcdef123456` | Intentional FAKE fixture for the redaction test |
| `tests/__pycache__/test_gateway*.pyc` | `sk-SECRET1234567`, `sk_live_abcdef123456` | Compiled cache of the same fake fixtures; gitignored |

No real API keys were found.

### Secret files
No `.env`, `secrets.toml`, `*.pem`, or `*.key` files exist on disk (outside the
vendored `.venv`). `.env.example` is a template with all key values blank.

### .gitignore coverage
Confirmed ignored: `.env`, `.streamlit/secrets.toml`, `__pycache__/`, `.venv/`,
`.cache/`.

## Keys to rotate

**None.** No real credential was ever committed (there is no history) or is present in
the tree. If a real key was ever pasted into a shell, an editor, or a chat outside
this repo, rotate that key as a precaution, but nothing in this project requires it.

## Preventive controls added

- `.gitleaks.toml`: gitleaks config extending the default ruleset, allowlisting the
  known test fixtures and non-source paths (`.venv`, `.cache`, `*.pyc`, `Yelp-JSON`,
  `*.html`, `*.ipynb`, `dist-info/RECORD`).
- `.pre-commit-config.yaml`: gitleaks + detect-secrets + `detect-private-key` +
  large-file guard, run on every commit.
- `.secrets.baseline`: detect-secrets baseline (empty results; regenerate with
  `detect-secrets scan > .secrets.baseline`).
- `.github/workflows/ci.yml`: a `tests` job (pytest on Python 3.11) and a `gitleaks`
  job (full-history scan, `fetch-depth: 0`) on every push and pull request.
- `.env.example`: added `OPENAI_API_KEY` (the gateway reads it) so no real key is
  improvised into a tracked file.

## Manual steps for the maintainer

1. Initialize version control and verify ignores BEFORE the first commit:
   ```
   git init
   git add .gitignore .gitleaks.toml .pre-commit-config.yaml .secrets.baseline
   git status            # confirm .env, .venv/, .cache/ are NOT listed
   ```
2. Install the hooks:
   ```
   pip install pre-commit detect-secrets
   pre-commit install
   detect-secrets scan > .secrets.baseline   # regenerate a real baseline
   pre-commit run --all-files                 # first full pass
   ```
3. First commit and push to a new branch:
   ```
   git add -A
   git commit -m "Initial commit with secret-scanning controls"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   The CI workflow runs pytest and gitleaks automatically on the push.

## If a real key is ever committed later

1. Rotate the exposed key immediately at the provider (OpenAI, NVIDIA, or ASU
   Voyager), the removal below does not un-expose an already-pushed secret.
2. Scrub it from history:
   ```
   pip install git-filter-repo
   git filter-repo --replace-text <(echo "REGEX==>***") --force
   # or remove a whole file from all history:
   git filter-repo --path path/to/leaked.env --invert-paths --force
   ```
3. Force-push the rewritten history and have collaborators re-clone:
   ```
   git push --force --all
   git push --force --tags
   ```
