# Headline results

_Headline metrics are generated from `artifacts/model_metrics.parquet`._

Run the pipeline (`pipeline/04_features.py` then `05_survival.py` and `05b_tier1_survival.py`) and re-run `python scripts/generate_metrics_table.py` to populate this table with the C index, time-dependent AUC, and integrated Brier score, each with 95% confidence intervals. We never hand-type these numbers.
