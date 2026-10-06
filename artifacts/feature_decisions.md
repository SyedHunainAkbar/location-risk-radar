# Feature decisions (stage 05 Cox design matrix)

We clean the production Cox design matrix (full feature set, pooled) to keep the Newton-Raphson fit well conditioned. Every drop or transform is listed below with its reason. We never silently drop features.

| Feature | Action | Reason |
|---|---|---|
| theme_value | drop | zero variance (constant column); singular in the Cox Hessian |
| theme_other | drop | zero variance (constant column); singular in the Cox Hessian |
| review_vol_6m | drop | |r|=0.984 with review_vol_12m (> 0.95); kept more interpretable review_vol_12m |

Final design matrix: 1545 rows x 25 columns.

Topic collapse: the ~68 raw `topic_*` per-topic share columns are replaced by ~9 complaint-theme shares. Each BERTopic topic is matched to its nearest zero-shot theme (service_speed, staff_attitude, order_accuracy, food_quality, cleanliness, value, management_response, drive_thru_takeout, other) by cosine similarity between the MiniLM embedding of the topic's top words and each theme phrase, then shares are summed per theme. The full topic-to-theme assignment is in `artifacts/topic_theme_map.csv`.

Fit: ridge-penalized `CoxPHFitter(l1_ratio=0)`, penalizer chosen by cross-validated C index over [0.01, 0.05, 0.1, 0.5], with a damped `step_size=0.5` retry if Newton-Raphson returns a NaN step. Tier 1 uses subsampled inner tuning (one 3-fold CV on a stratified 10,000-row subsample of each outer training fold) for speed at the ~46k-restaurant scale.
