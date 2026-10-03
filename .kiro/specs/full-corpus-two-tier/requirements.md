# Requirements: Full-Corpus Two-Tier Modeling

## Introduction

Our proposal and EDA Phase A use the entire Yelp corpus (about 150K businesses and
7M reviews), but modeling so far used only the 30-chain cohort. We add a corpus tier
without breaking the cohort design. Tier 1 learns closure risk across all eligible
restaurants from engagement and sentiment aggregates; Tier 2 keeps the cluster-
specific cohort models and stacks the out-of-fold Tier 1 score as a feature. All
corpus processing streams in chunks so we never hold 7M review texts in memory.

The cohort contract is unchanged: 30 chains, 15 Fast Food and 15 Non-Fast Food,
cluster-specific Tier 2 models, and the same landmark design.

## Requirements

### Requirement 1: Streaming corpus ingestion

**User story:** As an analyst, I want a restaurant universe table and a monthly
activity panel over all restaurants, built in one memory-safe pass.

#### Acceptance Criteria
1. The system SHALL write `data/all_restaurants.parquet`: every business whose
   categories contain "Restaurants", with normalized `chain`, `cluster` (same Fast
   Food rule), and an `is_cohort` flag.
2. The system SHALL write `data/activity_monthly.parquet` with
   `business_id, month, n_reviews, mean_stars, n_tips, n_checkins`.
3. The panel SHALL be computed in a single streaming pass over the full review, tip,
   and check-in files using chunked processing. The system SHALL NEVER load all 7M
   review texts into memory at once.
4. The system SHALL log peak memory and wall-clock runtime for the pass.
5. Only restaurant businesses SHALL appear in the panel.

### Requirement 2: Corpus sentiment scorer and transfer test

**User story:** As a modeler, I want a sentiment scorer trained off the cohort and
proven to transfer to it.

#### Acceptance Criteria
1. The TF-IDF + calibrated linear scorer SHALL be trained on a stratified sample of
   up to 1,000,000 reviews drawn from NON-cohort restaurants, using a business-
   grouped split.
2. The system SHALL evaluate on (a) a held-out non-cohort test set and (b) all cohort
   reviews as an out-of-sample transfer test, each with bootstrap 95% CIs.
3. The system SHALL score all cohort reviews fully.
4. The system SHALL compute monthly sentiment aggregates for ALL restaurants by
   scoring reviews in streaming batches, storing only aggregates, never text.

### Requirement 3: Tier 1 corpus survival model

**User story:** As a modeler, I want a corpus-wide closure-risk model under the same
landmark design.

#### Acceptance Criteria
1. Tier 1 SHALL use the same landmark design: T = 2018-01-01 primary and 2019-01-01
   sensitivity, a 24-month horizon, and features drawn strictly before T.
2. Tier 1 features SHALL be engagement, check-in, tip, stars, and sentiment
   aggregates. Tier 1 SHALL NOT use topic features.
3. Tier 1 SHALL fit Cox PH and an XGBoost model (`survival:cox` or a 24-month
   classifier); LightGBM is optional.
4. Validation SHALL use GroupKFold by chain for chained businesses and by business
   otherwise, plus temporal validation.
5. Tier 1 SHALL reproduce the Phase A headline as an ablation: stars-only vs
   engagement vs engagement + sentiment, with C index and time-dependent AUC and CIs.

### Requirement 4: Tier 2 cohort stacking (no leakage)

**User story:** As a modeler, I want the cohort models to benefit from the corpus
signal without leaking.

#### Acceptance Criteria
1. Tier 2 SHALL keep the cluster-specific cohort models.
2. Tier 2 SHALL add the Tier 1 risk score as a feature, computed out of fold: a
   cohort location's Tier 1 prediction SHALL come from Tier 1 folds that excluded it.
3. The system SHALL report whether Tier 2 beats Tier 1 on the cohort and whether
   topic features add lift, as an ablation table.

### Requirement 5: Updated risk scores, model card, tests

#### Acceptance Criteria
1. `artifacts/risk_scores.parquet` SHALL add `tier1_score`, `tier2_score`, and
   `final_score`, with a documented choice rule for `final_score`.
2. The model card SHALL document the two-tier design, the transfer test, the stacking
   anti-leakage rule, and the final-score rule.
3. Tests SHALL assert streaming monthly aggregates match a brute-force computation on
   a small fixture, and that stacking uses only out-of-fold Tier 1 predictions.

### Requirement 6: Notebook Phase A and Phase B

#### Acceptance Criteria
1. Notebook sections 2 and 6 SHALL present Phase A (corpus) and Phase B (cohort)
   results side by side, matching the EDA's two-phase structure.

### Requirement 7: Determinism, memory safety, CLI

#### Acceptance Criteria
1. All randomness SHALL use `config.SEED`; all paths SHALL come from `config.py`.
2. New pipeline scripts SHALL accept CLI args, cap memory via chunking, and be
   idempotent.
3. The corpus pipelines SHALL remain runnable offline; the app still reads only the
   small artifacts.
