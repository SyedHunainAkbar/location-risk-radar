# Data dictionary

Every artifact the app and notebook read. Large artifacts (reviews, embeddings) live
under `data/` and are gitignored; small outputs the app needs live under `artifacts/`
and are committed, each under 50 MB.

## artifacts/cohort_locations.parquet

| Column | Type | Meaning |
|---|---|---|
| business_id | str | Yelp business id |
| chain | str | normalized chain display name |
| cluster | str | Fast Food or Non-Fast Food |
| city, state | str | location |
| lat, lon | float | coordinates |
| stars | float | business average stars |
| review_count | int | business review count |
| is_open | int | 1 open, 0 closed (status, not a date) |

## data/cohort_reviews.parquet

`review_id, business_id, stars, date, text, useful, funny, cool`. Cohort reviews
only. Gitignored (can exceed 50 MB).

## data/activity_monthly.parquet (corpus panel)

`business_id, month, n_reviews, mean_stars, n_tips, n_checkins`. One row per
business-month over all restaurants.

## data/all_restaurants.parquet

`business_id, name, chain, cluster, is_cohort, city, state, stars, review_count,
is_open, latitude, longitude`.

## artifacts/model_metrics.parquet

`tier, scope, feature_set, model, harrell_c, harrell_c_lo, harrell_c_hi, uno_c,
auc_12m, auc_24m, ibs`. Tier 1 and Tier 2 ablation rows with 95% CIs.

## artifacts/risk_scores.parquet

`business_id, chain, cluster, risk_score, risk_percentile, risk_tier, top_3_drivers,
surv_12m, surv_24m, tier1_score, tier2_score, final_score`. `risk_tier` is
High / Elevated / Watch / Low.

## artifacts/survival_curves.parquet

`business_id, t_days, survival`. Per-location predicted survival over the horizon.

## artifacts/sentiment_benchmark.parquet

`model, features, accuracy, accuracy_lo, accuracy_hi, macro_f1, macro_f1_lo,
macro_f1_hi, recall_neg, recall_pos, roc_auc, roc_auc_lo, roc_auc_hi, n_test`.
LA2 classical and LA3 deep models.

## artifacts/sentiment_transfer_eval.parquet

The benchmark schema plus a `split` column (`noncohort_test`, `cohort_transfer`).

## data/review_sentiment.parquet

`review_id, business_id, p_positive`. Calibrated probability of positive sentiment
for every cohort review.

## artifacts/topics_{cluster}.parquet

`topic, label, top_words, size`. One file per cluster.

## data/topic_shares.parquet

Per business, per pre-landmark window: `business_id`, `cluster`, and one
`topic_{id}` share column per topic. A survival feature.

## artifacts/briefs.parquet

`business_id, tier, drivers, evidence, confidence, confidence_label, actions,
groundedness, n_evidence, auditor_grade, n_rejected_claims`. Precomputed Location
Risk Briefs for High and Elevated locations.

## artifacts/agent_eval.parquet

`n_briefs, numeric_fidelity_rate, mean_groundedness, auditor_rejection_rate,
mean_latency_s, total_est_cost_usd, confidence_formula`.

## artifacts/rag_cache.parquet

`business_id, question, answer, visible_answer, groundedness_rate, n_citations`.
Precomputed answers for the six standard questions per high-risk location.

## data/rag_index/

`rag_embeddings.npy` (float16) and `rag_metadata.parquet`
(`chunk_id, business_id, chain, review_id, date, stars, text`). The whole index stays
under the 50 MB cap. Gitignored.

## artifacts/gateway_models.json

Verified model ids per provider, substitutions, and a check timestamp.

## artifacts/llm_ledger.parquet

Per-call usage: `role, provider, model, latency_s, prompt_tokens, completion_tokens,
est_cost_usd, cache_hit, fallback_used`. No secrets.
