# Requirements: Landmark Survival

## Introduction

We estimate each cohort location's closure risk with a landmark survival design.
Features use only data dated strictly before the landmark T; the outcome is observed
over the 24 months after T. We model each cluster separately and pooled with a
cluster indicator, benchmark Cox PH, Random Survival Forest, and an XGBoost 24-month
classifier, validate with grouped and temporal splits, prove the value of text over
stars with an ablation, and emit calibrated risk scores plus a model card. This is
pipeline stages 04 (features) and 05 (survival).

## Requirements

### Requirement 1: Landmark population

#### Acceptance Criteria
1. The landmark T SHALL be 2018-01-01 (primary) with 2019-01-01 as a sensitivity
   landmark.
2. The population SHALL be cohort locations whose first review is before T minus 12
   months AND that have any activity (review, tip, or check-in) in the 6 months
   before T.
3. Population selection SHALL use only data dated before T.

### Requirement 2: Event and duration (proxy closure)

#### Acceptance Criteria
1. The proxy closure date SHALL be the location's last activity date across reviews,
   tips, and check-ins.
2. The event indicator SHALL be 1 when `is_open == 0` AND the proxy closure date is
   in the interval (T, T + 24 months]; otherwise 0 (right censored).
3. The duration SHALL be days from T to `min(proxy closure date, T + 24 months,
   DATASET_END)`.
4. Locations without an event in the window SHALL be right censored at that duration.

### Requirement 3: Leakage-safe features (only data dated < T)

#### Acceptance Criteria
1. Every feature SHALL be computed from records dated strictly before T.
2. The feature set SHALL include: review volume (12m, 6m, 3m); review velocity slope
   (OLS on 24 monthly counts); velocity ratio (last 6m / prior 6m); check-in volume
   and slope; tip volume and coverage; average stars all-time and last 12m; stars
   trend; text sentiment mean and slope from the production scorer; share of negative
   reviews; topic shares; LA1 lexicon complaint rate; review length; location age;
   chain-level priors computed out of fold; cluster.
3. The feature set SHALL deliberately EXCLUDE "days since last activity measured at T
   or later" and any post-T signal.
4. Chain-level priors SHALL be computed out of fold to avoid target leakage.
5. A leakage test SHALL assert no feature row uses any record dated on or after T.

### Requirement 4: Models

#### Acceptance Criteria
1. Four nested feature sets SHALL be modeled: (1) stars-only baseline, (2)
   engagement-only, (3) engagement + text, (4) full.
2. Each SHALL be fit per cluster and pooled with a cluster indicator.
3. Each SHALL be fit as Cox PH (lifelines, penalizer tuned), Random Survival Forest
   (scikit-survival), and an XGBoost classifier on the 24-month event for comparison.

### Requirement 5: Validation and metrics

#### Acceptance Criteria
1. Validation SHALL use GroupKFold by chain (5 folds) AND a temporal check (train on
   T=2018, test on T=2019).
2. Metrics SHALL include Harrell C index, Uno C index, time-dependent AUC at 12 and
   24 months, and integrated Brier score, each with bootstrap 95% CIs.
3. A calibration curve SHALL be produced.
4. An ablation table SHALL quantify the value of text over stars.
5. The proportional-hazards assumption SHALL be tested (Schoenfeld) for Cox, and the
   handling of any violations SHALL be stated.

### Requirement 6: Explainability

#### Acceptance Criteria
1. Cox hazard ratios with confidence intervals SHALL be reported.
2. SHAP values SHALL be computed for the XGBoost model.
3. The top drivers per location SHALL be produced.

### Requirement 7: Outputs

#### Acceptance Criteria
1. `artifacts/risk_scores.parquet` SHALL contain `business_id, chain, cluster,
   risk_score, risk_percentile, risk_tier` (High/Elevated/Watch/Low), `top_3_drivers`,
   and predicted 12-month and 24-month survival.
2. `artifacts/model_metrics.parquet` SHALL contain the benchmark and ablation metrics.
3. `artifacts/survival_curves.parquet` SHALL contain per-location survival curves for
   the app.
4. A model card markdown SHALL be written documenting design, metrics, PH handling,
   and limitations.

### Requirement 8: Determinism, idempotency, CLI

#### Acceptance Criteria
1. All randomness SHALL use `config.SEED`; all paths SHALL come from `config.py`.
2. `pipeline/04_features.py` and `pipeline/05_survival.py` SHALL accept CLI args and
   be idempotent.

### Requirement 9: Tests

#### Acceptance Criteria
1. A leakage test SHALL assert no feature row uses data dated on or after T.
2. Event and duration definitions SHALL be verified on a synthetic timeline covering
   closed-in-window, closed-after-window, open, and censored-at-DATASET_END cases.
3. Risk tiering and out-of-fold chain priors SHALL be tested.
4. Tests SHALL run without scikit-survival, lifelines, xgboost, or shap installed
   (the heavy modeling is exercised offline; unit tests cover the pure logic).
