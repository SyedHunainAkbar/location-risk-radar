# Design: Full-Corpus Two-Tier Modeling

## Overview

Two tiers over the same landmark design. Tier 1 is a corpus-wide closure-risk model
trained on all eligible restaurants from engagement and sentiment aggregates. Tier 2
is the existing cluster-specific cohort model, now stacked with the out-of-fold Tier 1
risk score. Everything that touches the 7M-review file streams in chunks and stores
only aggregates.

New module `src/lrr/corpus.py` holds the streaming pass and the panel-to-feature
reducers. `sentiment.py` gains non-cohort sampling, a transfer evaluation, and a
streaming batch scorer. `features.py` and `survival.py` gain Tier 1 feature assembly
from the monthly panel and the out-of-fold stacking helper. New pipeline stages
`01b`, `02c`, `04b`, `05b`, `05c` drive the work; the online app is unchanged.

Heavy modeling libraries stay lazy. The streaming pass uses pandas chunked
`read_json`/`iter_yelp_records` (polars optional) and never materializes review text
beyond the current chunk.

## corpus.py

```python
def build_all_restaurants(business_df, cohort_business_ids) -> DataFrame
def stream_activity_panel(reviews_iter, tips_iter, checkins, restaurant_ids,
                          chunksize) -> DataFrame     # business_id, month, n_reviews, mean_stars, n_tips, n_checkins
def accumulate_review_chunk(acc, chunk, restaurant_ids) -> None   # updates running sums
def finalize_panel(acc) -> DataFrame
def log_resources(label, start_time, start_rss) -> None
def panel_window_features(panel, T, windows) -> DataFrame         # pre-T aggregates per business
```

- `build_all_restaurants`: filter businesses whose categories contain "Restaurants",
  attach the canonical `chain` and `cluster` from `cohort.py`, and set
  `is_cohort = business_id in cohort_business_ids`.
- `stream_activity_panel`: one pass. For reviews we read in chunks and accumulate per
  (business_id, month): review count and the running sum of stars (to derive
  `mean_stars` at finalize). Tips accumulate per month count. Check-ins (already
  exploded, far smaller) accumulate per month count. Only restaurant business ids are
  kept. We hold a dict/DataFrame of aggregates, never the raw text.
- `log_resources`: logs wall-clock seconds and peak RSS via `resource` (POSIX) or
  `psutil`/`tracemalloc` fallback, so the one-pass memory claim is auditable.
- `panel_window_features`: from the monthly panel, derive strictly-pre-T window
  aggregates (volume 12/6/3m, slope, tip and check-in volume, mean stars and trend)
  per business. These are the Tier 1 features; no text, no topics.

## sentiment.py additions

```python
def noncohort_training_sample(reviews, all_restaurants, n=1_000_000, seed=SEED)
def transfer_evaluate(scorer, noncohort_test, cohort_reviews) -> dict   # (a) and (b) with CIs
def score_reviews_streaming(reviews_iter, scorer, restaurant_ids, chunksize) -> DataFrame
                                                                 # business_id, month, mean_p_positive, n
```

- `noncohort_training_sample` draws up to 1,000,000 labeled (1-2 vs 4-5 star) reviews
  from businesses with `is_cohort == False`, then a business-grouped split.
- `transfer_evaluate` reports the benchmark metric set with bootstrap CIs on the
  held-out non-cohort test set and, separately, on all cohort reviews as the transfer
  test.
- `score_reviews_streaming` applies the calibrated scorer in chunks and reduces to
  monthly `mean_p_positive` and `n` per business, storing only aggregates.

## features.py / survival.py additions

```python
# features.py
def tier1_features(all_restaurants, panel, sentiment_monthly, T) -> DataFrame
def tier1_labels(all_restaurants, last_activity, T) -> DataFrame        # reuse event/duration logic
def group_key(df) -> Series            # chain when chained, else business_id

# survival.py
def fit_tier1(features, labels) -> model                                # Cox + XGBoost
def oof_tier1_scores(features, labels, group, n_folds) -> Series        # out-of-fold risk per row
def final_score(tier1, tier2) -> (score, rule)                          # documented choice rule
FINAL_SCORE_RULE: str
```

- Tier 1 reuses the existing `label_event_duration` logic (proxy closure, event in
  (T, T+24m], duration cap). Groups are the chain for chained businesses and the
  business id otherwise, so grouped CV never leaks a chain across folds while singleton
  restaurants still group by themselves.
- `oof_tier1_scores`: fit Tier 1 on all-but-one fold, predict the held-out fold,
  repeat; the cohort rows receive Tier 1 scores only from folds that excluded them.
  These scores become the `tier1_score` stacking feature for Tier 2.
- `final_score`: documented rule. For cohort locations use `tier2_score` (the cluster-
  specific model, which has seen the stacked corpus signal); for non-cohort locations
  use `tier1_score`. The rule is stored in `FINAL_SCORE_RULE` and written to the card.

## Stacking anti-leakage (the core correctness property)

A cohort location L in chain C must never receive a Tier 1 score from a Tier 1 model
that was trained on L (or on C, since Tier 2 groups by chain). `oof_tier1_scores`
assigns folds by `group_key` so that all rows sharing a group land in the same held-
out fold; the Tier 1 model for that fold is trained on the complement. The test
asserts, for every cohort row, that its group did not appear in the training folds
that produced its score.

## Pipelines

- `01b_corpus_ingest.py`: build `all_restaurants.parquet` and `activity_monthly.parquet`
  with resource logging.
- `02c_corpus_sentiment.py`: train the non-cohort scorer, run the transfer test, score
  cohort reviews, and stream monthly sentiment aggregates for all restaurants.
- `04b_tier1_features.py`: assemble Tier 1 features and labels per landmark.
- `05b_tier1_survival.py`: fit Tier 1, validate (grouped + temporal), write the Phase A
  ablation into `model_metrics.parquet`, and emit out-of-fold Tier 1 scores.
- `05c_tier2_stack.py`: add the out-of-fold `tier1_score` to the cohort features,
  refit Tier 2, run the Tier2-vs-Tier1 and topic-lift ablations, update
  `risk_scores.parquet` with `tier1_score/tier2_score/final_score`, and refresh the
  model card.

## Testing strategy

Unit tests, no heavy libs, no real data:

- `stream_activity_panel` on a tiny fixture equals a brute-force groupby
  (`n_reviews`, `mean_stars`, `n_tips`, `n_checkins` per business-month).
- `build_all_restaurants` sets `is_cohort` correctly and keeps only restaurants.
- `oof_tier1_scores` / fold assignment: for each row, its group is absent from the
  training folds that produced its score (the anti-leakage invariant).
- `final_score` applies the documented rule (cohort -> tier2, else tier1).
- `score_reviews_streaming` reduces to the same monthly means as a brute-force pass.
