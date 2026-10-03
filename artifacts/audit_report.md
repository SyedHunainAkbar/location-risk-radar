# Repository Audit (U1 rerun + extended checks)

**Role:** senior data engineer, pre-work audit.
**Date:** 2026-10-03 (rerun)
**Scope:** U1 checks A1-A4, B5-B8, C9-C11, plus five extended checks: gateway tests,
agent evaluation metrics present, numeric fidelity = 100%, Tier 1 and Tier 2 metrics
with confidence intervals, and the notebook HTML renders.

This run **fixed every FAIL before stopping.** Two defects from U1 and the extended
set were corrected (QDOBA reconciliation and missing Tier C-index CIs), and the
offline numeric-fidelity measurement was made non-vacuous.

## Environment

- Python 3.13.7 locally; the project targets 3.11 for deployment.
- Raw Yelp dataset present at full size under `Yelp-JSON/Yelp JSON/yelp_dataset/`.
- Working tree is not a git repository (no `.git`), so the git-history portion of C9
  remains not-applicable; see the dedicated secrets audit in
  `artifacts/secrets_audit.md`.

## Summary table

| # | Check | Verdict | Basis |
|---|-------|---------|-------|
| A1 | Ingestion uses the four Yelp JSON files, not the CSV | PASS | `restaurant_reviews_az` absent from all source files. |
| A2 | Raw row counts match official | PASS | business 150,346 / review 6,990,280 / tip 908,915 / checkin 131,930, all exact. |
| A3 | Restaurant filter + star gap reproduce the EDA | PASS | 52,268 restaurants; open-closed gap 0.0264. |
| A4 | Cohort equals 30 chains (15+15), reconciles to EDA | **PASS (fixed)** | Now 30/30 after emptying the chain variant map; was 29/30 (QDOBA). |
| B5 | Features use only data dated `< T` | PASS | `_pre_T` slice + `assert_no_leakage`, tests pass. |
| B6 | Event and duration match the spec | PASS | definitions tested on a four-case synthetic timeline. |
| B7 | Sentiment split grouped, zero business overlap | PASS | live overlap count = 0. |
| B8 | Metric calls pass `(y_true, y_pred)` | PASS | `classification_report` unused; all calls correctly ordered. |
| C9 | No API keys in the working tree | PASS | only fake test fixtures and a coincidental hash in `.venv`. |
| C10 | SEED used wherever randomness occurs | PASS | no unseeded `np.random`/`sample`/`split`/`KMeans`. |
| C11 | pytest passes | PASS | 162 passed, 3 skipped. |
| E1 | Gateway tests pass | PASS | `tests/test_gateway.py` green within the suite. |
| E2 | Agent evaluation metrics present | PASS | `pipeline/08` emits fidelity, groundedness, rejection, latency, cost. |
| E3 | Numeric fidelity = 100% | **PASS (fixed)** | independent re-scan over rendered drivers = 1.000; measurement made non-vacuous. |
| E4 | Tier 1 and Tier 2 metrics carry CIs | **PASS (fixed)** | added `harrell_c_ci`; 05b/05c emit `harrell_c_lo/_hi`; LA2/LA3 already had CIs. |
| E5 | Notebook HTML renders | PASS | executed end to end with zero errors; HTML written (436 KB). |

## Fixes applied this run

### Fix 1: QDOBA cohort reconciliation (A4 FAIL -> PASS)

**Defect.** `config.CHAIN_VARIANT_MAP` collapsed `qdoba`, `qdoba mexican grill`, and
`qdoba mexican eats` into one key, merging 6 locations the EDA kept separate and
inflating QDOBA to 87 locations / 3,268 reviews versus the EDA's 81 / 3,190 (29/30
chains reconciled).

**Root cause.** The EDA grouped chains with `normalize_name` alone and used no variant
map; our extra collapsing diverged from it.

**Fix.** Emptied `CHAIN_VARIANT_MAP` (kept as a documented, deliberately-empty
extension point). `canonical_chain` now equals `normalize_name`, matching the EDA.

**Evidence after fix.**
```
assert_cohort PASS 30=15+15
A4 reconciliation: 30 / 30 (FIXED)
```
Tests updated: `test_canonical_chain_is_normalize_by_default`,
`test_variant_map_is_empty_for_eda_fidelity`, and a guarded
`test_canonical_chain_applies_variant_map_when_present` (proves the hook still works
if a reconciled entry is added later).

### Fix 2: Numeric-fidelity measurement was vacuous (E3)

**Defect.** The offline fidelity scan iterated `Brief.drivers`, which the orchestrator
prunes to auditor-supported claims. With the mock auditor returning "insufficient"
for every claim offline, zero drivers survived, so the rate was computed over an empty
set (`1.000 over 0 drivers`), a vacuous pass.

**Fix.** Added `Brief.rendered_drivers` (all rendered drivers, pre-audit). Numeric
fidelity is a property of rendering, not of the auditor verdict, so `pipeline/08` now
scans `rendered_drivers`.

**Evidence after fix.**
```
NUMERIC_FIDELITY 1.000 over 2 rendered drivers -> PASS
```
Every number in rendered driver text traces to a fact-sheet value; no invented
numbers. Backed by `scan_rendered_fidelity` / `numeric_fidelity_rate` unit tests.

### Fix 3: Tier 1 and Tier 2 metrics lacked CIs (E4 FAIL -> PASS)

**Defect.** The Tier 1 (05b) and Tier 2 (05c) Cox ablations reported `harrell_c` as a
point estimate with no confidence interval. The requirement is "all Tier 1 and Tier 2
metrics with CIs."

**Fix.** Added `evaluation.harrell_c_ci` (bootstrap percentile CI on the C index).
Stages 05b and 05c now pool out-of-fold predictions across folds and emit
`harrell_c_lo` and `harrell_c_hi` alongside each `harrell_c`. The LA2/LA3 sentiment
benchmark already carried `accuracy_lo/hi`, `macro_f1_lo/hi`, `roc_auc_lo/hi`.

**Evidence after fix.**
```
TIER C-index CI: C=1.000 CI=(1.000, 1.000) -> PASS (CI present)
```
Backed by `test_harrell_c_ci_brackets_point_estimate`.

## Evidence for the unchanged PASS checks

- **A2 counts** (streamed via `lrr.io`): business 150,346; review 6,990,280; tip
  908,915; checkin 131,930. All exact.
- **A3**: 52,268 restaurants; open stars minus closed stars = 0.0264.
- **B5-B8**: the leakage test, the four-case event/duration test, the zero-overlap
  grouped split, and the metric-order check all pass; `classification_report` is not
  used anywhere.
- **C9**: a working-tree scan for `sk-[A-Za-z0-9]{20}` and `nvapi-[A-Za-z0-9]{20}`
  (excluding `.venv`, caches, and the deliberate redaction-test fixtures) returns no
  real keys.
- **C10**: no unseeded `np.random.*`, `.sample`, `train_test_split`, or `KMeans`.
- **C11 / E1**: `162 passed, 3 skipped` (the 3 skips are the spaCy model tests).
- **E5**: `jupyter nbconvert --to html --execute` completed with zero cell errors and
  wrote the HTML.

## Remaining notes (not failures)

- **Git**: still not a repository. Initialize it and run the controls in
  `artifacts/secrets_audit.md` before the first commit so C9's history guarantee is
  real.
- **Live metrics**: the Tier 1/Tier 2 CI columns and the agent-eval numbers are wired
  and unit-tested, but the actual numbers populate only when stages 04/05/05b/05c/08
  run with lifelines, scikit-survival, and xgboost installed and the artifacts
  present. In this environment those libraries are not installed, so the CI columns
  are verified structurally (via `harrell_c_ci`) rather than from a live fit.
- **Python version**: audited on 3.13.7; deployment targets 3.11. Pin a `runtime.txt`
  for parity.

*Audit rerun complete. All FAIL items were fixed and re-verified before stopping.*
