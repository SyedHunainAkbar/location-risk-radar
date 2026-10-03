# Design: Cohort Ingestion

## Overview

Stage 01 turns raw Yelp JSON into cohort-scoped parquet artifacts and a data quality
report. The logic mirrors the EDA notebook exactly so the cohort is reproducible,
and the result is asserted against a locked constant list in `config.py`.

Three modules carry the work:

- `src/lrr/io.py`: streaming readers for `.tar` and `.json`, plus parquet helpers.
- `src/lrr/cohort.py`: pure functions for normalization, labeling, selection,
  assertion, and the data quality report.
- `pipeline/01_ingest_cohort.py`: the idempotent CLI that wires it together.

## io.py

```python
def iter_yelp_records(dataset: str, yelp_dir: Path) -> Iterator[dict]: ...
def read_yelp_frame(dataset: str, yelp_dir: Path, columns, chunksize) -> DataFrame: ...
def write_parquet(df: DataFrame, path: Path) -> Path: ...
def read_parquet(path: Path) -> DataFrame: ...
```

Resolution order per Requirement 1: prefer `yelp_academic_dataset_<dataset>.json` in
`YELP_DIR`; else find a `.tar` in `YELP_DIR` and stream the matching member; else
raise `FileNotFoundError`. Streaming uses a line generator so memory stays flat;
`.tar` members are read through the archive's file object. `read_yelp_frame` batches
records into DataFrames to bound memory on the 5 GB review file.

## cohort.py

Pure, testable functions:

```python
def normalize_name(name: str) -> str
def canonical_chain(name: str) -> str              # normalize + variant map
def is_restaurant(categories: str) -> bool
def is_cafe(categories: str) -> bool
def is_fast_food_location(categories: str) -> bool
def clean_business(df) -> DataFrame
def build_chain_summary(rest_df) -> DataFrame
def select_cohort(chain_summary) -> DataFrame      # 15 + 15 by review count
def assert_cohort(cohort) -> None                  # 30 = 15 + 15 and name match
def reconciliation_table(cohort) -> DataFrame      # vs config EDA values
def data_quality_report(...) -> dict
```

`normalize_name` is the EDA function verbatim:
`re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(s).lower().strip()))`.
`canonical_chain` applies `normalize_name` then a small `CHAIN_VARIANT_MAP` from
`config.py` so spelling variants (for example "Chick-Fil-A" and "Chick-fil-A", both
normalizing to `chickfila`) collapse to one canonical key.

Selection reproduces the EDA: group by canonical chain, aggregate average stars,
unique location count, summed review count, and a Fast Food max flag; keep chains
with stars > 2.5 and locations > 3; sort by total review count descending; take 15
per cluster. `assert_cohort` checks the 15/15 split and set-equality of display
names against `config.COHORT_FAST_FOOD` and `config.COHORT_NON_FAST_FOOD`.

## pipeline/01_ingest_cohort.py

Steps: read and clean business, filter restaurants, exclude cafes, build chain
summary, select and assert cohort, scope reviews/tips/checkins to cohort business
ids, write artifacts, print the reconciliation table and the data quality report.
CLI flags: `--yelp-dir`, `--artifacts-dir`, `--data-dir`, `--dry-run`. Idempotent:
each run overwrites the same parquet files from the same inputs.

## Artifacts and schemas

- `artifacts/cohort_locations.parquet`: `business_id, chain, cluster, city, state,
  lat, lon, stars, review_count, is_open`.
- `data/cohort_reviews.parquet`: `review_id, business_id, stars, date, text, useful,
  funny, cool`.
- `data/cohort_tips.parquet`: `business_id, text, date, compliment_count,
  user_id` (as available).
- `data/cohort_checkins.parquet`: exploded `business_id, checkin_at`.

Reviews, tips, and checkins live under `data/` (gitignored, can exceed 50 MB).
`cohort_locations.parquet` is small and committed under `artifacts/`.

## Data quality report

A dict plus a printed summary: duplicate counts, null/empty review text count,
review date range, and the list and count of cohort locations with zero check-ins
and zero tips. Missing engagement is flagged, never imputed (Requirement 5.5).

## Testing strategy

A tiny synthetic business/review table built in-memory exercises normalization and
selection with no real data. We assert a correct 15/15 cohort on a crafted fixture
and assert the negative case (wrong split raises). Normalization tests cover case,
punctuation, trademark symbols, whitespace, and variant mapping.
