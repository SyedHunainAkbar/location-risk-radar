# Methodology

## Landmark survival design

We freeze a landmark date `T` (2018-01-01 primary, 2019-01-01 sensitivity), build
every feature from records dated strictly before `T`, and observe closure over the
next 24 months. The data gives a current status flag (`is_open`), not a closure
date, so we proxy the closure date with the last observed activity across reviews,
tips, and check-ins.

- **Event:** `is_open == 0` and the proxy closure date falls in `(T, T + 24 months]`.
- **Duration:** days from `T` to `min(proxy closure, T + 24 months, dataset end)`.
- **Censoring:** locations without an event in the window are right-censored.

## Leakage controls

- Every feature source is sliced to `date < T` before any aggregation.
- An explicit assertion fails the build if any feature row uses a record dated on or
  after `T`.
- The feature set deliberately excludes "days since last activity measured at or
  after `T`".
- Chain-level priors are computed out of fold, so a row's prior never sees its own
  fold's labels.
- Validation uses GroupKFold by chain (and by business for singletons), plus a
  temporal check that trains at the 2018 landmark and tests at 2019.

## Two-tier stacking

- **Tier 1** is a corpus-wide survival model over all eligible restaurants, using
  engagement, check-in, tip, stars, and sentiment aggregates. No topic features.
- **Tier 2** keeps the cluster-specific cohort models and adds the Tier 1 risk score
  as a feature. The Tier 1 score for each cohort location comes only from Tier 1
  folds that excluded its chain, so stacking adds the corpus signal without leakage.
- **Final score:** `tier2_score` for cohort locations (the calibrated cluster model),
  `tier1_score` for every other restaurant.

## Metrics

We report Harrell's C and Uno's C, time-dependent AUC at 12 and 24 months, and the
integrated Brier score, each with bootstrap 95% confidence intervals. The ablation
contrasts stars-only against engagement, engagement plus text, and the full set to
show the lift from engagement and language over the star rating. We run the
Schoenfeld test for proportional hazards and stratify or add a time interaction for
any violating covariate. All numbers are generated from the artifacts, never
hand-typed; see `scripts/generate_metrics_table.py`.
