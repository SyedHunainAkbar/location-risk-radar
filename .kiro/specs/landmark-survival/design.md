# Design: Landmark Survival

## Overview

A landmark design freezes a cutoff T, builds features from the strict past
(records dated < T), and observes closure over (T, T + 24 months]. We model each
cluster separately and pooled with a cluster indicator, compare Cox PH, Random
Survival Forest, and an XGBoost 24-month classifier, and emit calibrated risk
scores. Stages: 04 builds the leakage-safe feature matrix and labels; 05 trains,
validates, explains, and writes the app artifacts and model card.

The pure logic (population, event/duration, feature construction, metrics,
tiering) lives in importable modules and is unit-tested without any survival
library. lifelines, scikit-survival, xgboost, and shap are imported lazily inside
the model-fitting functions and the stage-05 driver.

## features.py

```python
def landmark_population(reviews, tips, checkins, locations, T) -> DataFrame
def last_activity_date(reviews, tips, checkins) -> Series            # per business
def label_event_duration(pop, last_activity, locations, T) -> DataFrame
def monthly_counts(dates, T, months) -> np.ndarray                   # pre-T only
def ols_slope(y) -> float                                            # velocity slope
def build_features(reviews, tips, checkins, locations, sentiment, shares, lexicon, T)
def out_of_fold_chain_priors(df, group="chain", fold="fold") -> Series
def assert_no_leakage(records, T) -> None                            # raises on >= T
```

- `landmark_population`: keep locations with first review < T - 12 months and any
  review/tip/check-in in [T - 6 months, T). All filters use dates < T.
- `last_activity_date`: max date across the three activity sources per business.
  This is the proxy closure date (Requirement 2.1).
- `label_event_duration`: event = `is_open == 0 and T < proxy <= T + 24m`; duration
  = days from T to `min(proxy, T + 24m, DATASET_END)`; censored otherwise.
- Feature builders operate on pre-T slices only. `monthly_counts` bins the 24 months
  before T; `ols_slope` is the least-squares slope of the monthly series. Velocity
  ratio is last-6m count / prior-6m count (guarded against divide-by-zero).
- `out_of_fold_chain_priors`: for each fold, the chain's mean event rate is computed
  on the other folds only, so the prior for a row never sees its own fold's labels.
- `assert_no_leakage`: given a long record frame with a `date` column, raises if any
  date is on or after T. Stage 04 calls this on every source slice.

## survival.py

```python
FEATURE_SETS: dict[str, list[str]]        # stars_only, engagement, eng_text, full
def design_matrix(features, feature_set, pooled) -> DataFrame
def fit_cox(X, penalizer) -> model                     # lifelines, lazy
def fit_rsf(X) -> model                                # scikit-survival, lazy
def fit_xgb(X, event) -> model                         # xgboost, lazy
def risk_tier(percentile) -> str                       # High/Elevated/Watch/Low
def top_drivers(contribs, feature_names, k=3) -> list
def build_risk_scores(models, features) -> DataFrame
def survival_curves(model, features) -> DataFrame
```

Feature sets are nested so the ablation is clean. `design_matrix` selects columns
and, when `pooled`, adds a `cluster_ff` indicator. Risk score is the model's risk
(Cox partial hazard, RSF risk, or XGBoost event probability), converted to a
percentile within the scored population; tiers cut the percentile at High >= 90,
Elevated >= 75, Watch >= 50, else Low. Drivers come from Cox hazard ratios or XGBoost
SHAP, top 3 by absolute contribution per location.

## evaluation.py

```python
def harrell_c(duration, event, risk) -> float
def uno_c(train_surv, test_surv, risk, tau) -> float          # scikit-survival, lazy
def time_dependent_auc(train, test, risk, times) -> dict      # 12, 24 months
def integrated_brier(train, test, surv_prob, times) -> float
def calibration_points(pred, observed, bins) -> DataFrame
def bootstrap_metric(fn, *arrays, n, ci, seed) -> (lo, hi)
def schoenfeld_ph_test(cox_model, X) -> DataFrame             # lifelines, lazy
def ablation_table(metrics_by_set) -> DataFrame               # text vs stars delta
```

Harrell's C is implemented directly (concordant/comparable pairs) so it needs no
library and is unit-testable. Uno C, time-dependent AUC, and IBS wrap
scikit-survival and are exercised offline. Bootstrap CIs resample the test rows.
`schoenfeld_ph_test` wraps lifelines `check_assumptions`/`proportional_hazard_test`;
violations are handled by stratifying the offending covariate or adding a
time-interaction, documented in the model card.

## Pipelines

- `pipeline/04_features.py`: load cohort artifacts, assert no leakage per source,
  build the population, labels, and feature matrix for a given T, write
  `data/features_T{year}.parquet`. CLI: `--landmark`, `--reviews`, `--locations`,
  `--tips`, `--checkins`, `--sentiment`, `--shares`, `--lexicon`, `--dry-run`.
- `pipeline/05_survival.py`: load features for the primary (and sensitivity) T, run
  the four feature sets x three model families per cluster and pooled, validate with
  GroupKFold(chain) and the temporal 2018->2019 check, compute metrics with bootstrap
  CIs, the ablation, PH test, and SHAP, then write `artifacts/risk_scores.parquet`,
  `artifacts/model_metrics.parquet`, `artifacts/survival_curves.parquet`, and
  `artifacts/survival_model_card.md`. CLI mirrors stage 04 plus `--provider` is not
  needed here.

## Testing strategy

Unit tests on synthetic timelines: event/duration across closed-in-window,
closed-after-window, open, and censored-at-DATASET_END; population inclusion rules;
`monthly_counts`/`ols_slope`; velocity ratio guards; `assert_no_leakage` raising on a
>= T record; `harrell_c` on a hand-checkable ranking; out-of-fold priors never seeing
their own fold; and risk tiering cut points. No lifelines, scikit-survival, xgboost,
or shap in the test path.
